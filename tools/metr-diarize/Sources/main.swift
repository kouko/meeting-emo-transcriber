import CoreML
import FluidAudio
import Foundation

// Voiceprint metadata constants (must match Go types.VoiceprintModel / VoiceprintProjection)
let voiceprintModel = "fluidaudio_embedding_v1"
let voiceprintProjection = "none"

@main
struct MetrDiarize {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            log("Usage: metr-diarize <audio.wav> [--threshold <float>] [--num-speakers <int>]")
            log("       metr-diarize --extract-voiceprints <file1.wav> [file2.wav ...]")
            exit(1)
        }

        // Mode: batch extract voiceprints (for enroll)
        if let idx = args.firstIndex(of: "--extract-voiceprints") {
            let wavFiles = Array(args[(idx + 1)...])
            guard !wavFiles.isEmpty else {
                log("Error: --extract-voiceprints requires at least one audio file")
                exit(1)
            }
            await extractVoiceprints(audioPaths: wavFiles)
            return
        }

        // Mode: backward compat aliases
        if let idx = args.firstIndex(of: "--extract-embeddings") {
            let wavFiles = Array(args[(idx + 1)...])
            await extractVoiceprints(audioPaths: wavFiles)
            return
        }
        if let idx = args.firstIndex(of: "--extract-embedding"), idx + 1 < args.count {
            await extractVoiceprints(audioPaths: [args[idx + 1]])
            return
        }

        // Mode: diarization (default)
        let audioPath = args[1]
        let threshold = parseFloat(flag: "--threshold", default: 0.6)
        let numSpeakers = parseInt(flag: "--num-speakers", default: 0)

        do {
            let (diarizer, _) = try await loadDiarizer(threshold: threshold, numSpeakers: numSpeakers)

            let speakersDesc = numSpeakers > 0 ? "\(numSpeakers)" : "auto"
            log("Running diarization (threshold=\(String(format: "%.2f", threshold)), num_speakers=\(speakersDesc))...")
            let url = URL(fileURLWithPath: audioPath)
            let result = try await diarizer.process(url)

            let speakerCount = Set(result.segments.map { $0.speakerId }).count
            log("Diarization complete: \(result.segments.count) segments, \(speakerCount) speakers")

            // Per-speaker clustering stats
            printClusteringStats(result: result)

            // Build segments output
            let segments = result.segments.map { seg -> [String: Any] in
                [
                    "start": Double(seg.startTimeSeconds),
                    "end": Double(seg.endTimeSeconds),
                    "speaker": seg.speakerId,
                ]
            }

            // Build speaker voiceprints (raw 256-dim WeSpeaker centroid)
            var speakerVoiceprints: [String: [Double]] = [:]
            if let db = result.speakerDatabase {
                for (speakerId, embedding) in db {
                    speakerVoiceprints[speakerId] = embedding.map { Double($0) }
                }
            }

            let json: [String: Any] = [
                "segments": segments,
                "speakers": speakerCount,
                "speaker_voiceprints": speakerVoiceprints,
            ]
            let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        } catch {
            log("Error: \(error)")
            exit(1)
        }
    }

    // MARK: - Batch Extract Voiceprints Mode

    static func extractVoiceprints(audioPaths: [String]) async {
        do {
            let (diarizer, _) = try await loadDiarizer(threshold: 0.6, numSpeakers: 1)

            var results: [[String: Any]] = []
            for audioPath in audioPaths {
                log("Extracting voiceprint: \(audioPath)...")
                let url = URL(fileURLWithPath: audioPath)
                let result = try await diarizer.process(url)

                guard let db = result.speakerDatabase, let firstEntry = db.first else {
                    log("Warning: no voiceprint extracted for \(audioPath)")
                    results.append([
                        "file": audioPath,
                        "vector": [] as [Double],
                        "dim": 0,
                        "model": voiceprintModel,
                        "projection": voiceprintProjection,
                    ])
                    continue
                }

                let embedding = firstEntry.value.map { Double($0) }
                results.append([
                    "file": audioPath,
                    "vector": embedding,
                    "dim": embedding.count,
                    "model": voiceprintModel,
                    "projection": voiceprintProjection,
                ])
            }

            log("\(results.count) voiceprints extracted")

            let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        } catch {
            log("Error: \(error)")
            exit(1)
        }
    }

    // MARK: - Shared Setup

    static func loadDiarizer(threshold: Float, numSpeakers: Int) async throws -> (OfflineDiarizerManager, OfflineDiarizerModels) {
        let modelsDir = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".metr/models/diarization")
        log("Loading diarization models from \(modelsDir.path)...")
        let models = try await OfflineDiarizerModels.load(from: modelsDir) { progress in
            log("  Downloading models: \(String(format: "%.0f", progress.fractionCompleted * 100))%")
        }

        var config = OfflineDiarizerConfig.default
        config.clustering.threshold = Double(threshold)
        if numSpeakers > 0 {
            config = config.withSpeakers(exactly: numSpeakers)
        }

        let diarizer = OfflineDiarizerManager(config: config)
        diarizer.initialize(models: models)
        return (diarizer, models)
    }

    // MARK: - Clustering Stats

    static func printClusteringStats(result: DiarizationResult) {
        // Group segments by speaker
        var speakerSegments: [String: (count: Int, duration: Float, totalQuality: Float)] = [:]
        for seg in result.segments {
            var stats = speakerSegments[seg.speakerId] ?? (0, 0, 0)
            stats.count += 1
            stats.duration += seg.durationSeconds
            stats.totalQuality += seg.qualityScore
            speakerSegments[seg.speakerId] = stats
        }

        let sortedSpeakers = speakerSegments.sorted { $0.value.duration > $1.value.duration }

        log("")
        log("Clustering summary:")
        for (speakerId, stats) in sortedSpeakers {
            let avgQuality = stats.count > 0 ? stats.totalQuality / Float(stats.count) : 0
            let durMin = String(format: "%.1f", stats.duration / 60)
            let qual = String(format: "%.2f", avgQuality)
            log("  \(speakerId): \(stats.count) segments, \(durMin) min (avg quality: \(qual))")
        }

        // Inter-cluster centroid similarity
        if let db = result.speakerDatabase, db.count > 1 {
            log("")
            log("Inter-cluster centroid similarity:")
            let speakerIds = db.keys.sorted()
            var closestPair: (String, String, Float) = ("", "", -1)

            for i in 0..<speakerIds.count {
                for j in (i+1)..<speakerIds.count {
                    let s1 = speakerIds[i], s2 = speakerIds[j]
                    if let e1 = db[s1], let e2 = db[s2] {
                        let sim = cosineSimilarity(e1, e2)
                        let simStr = String(format: "%.2f", sim)
                        log("  \(s1) vs \(s2): \(simStr)")
                        if sim > closestPair.2 {
                            closestPair = (s1, s2, sim)
                        }
                    }
                }
            }
            if closestPair.2 > 0 {
                let simStr = String(format: "%.2f", closestPair.2)
                log("  (closest pair: \(closestPair.0) vs \(closestPair.1) = \(simStr))")
            }
        }

        // Timing
        if let timings = result.timings {
            let seg = String(format: "%.1f", timings.segmentationSeconds)
            let emb = String(format: "%.1f", timings.embeddingExtractionSeconds)
            let cls = String(format: "%.1f", timings.speakerClusteringSeconds)
            let tot = String(format: "%.1f", timings.totalProcessingSeconds)
            log("\nTiming: segmentation=\(seg)s, embedding=\(emb)s, clustering=\(cls)s, total=\(tot)s")
        }
    }

    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<min(a.count, b.count) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    // MARK: - Helpers

    static func log(_ message: String) {
        FileHandle.standardError.write("[\(timestamp())] \(message)\n".data(using: .utf8)!)
    }

    static func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: Date())
    }

    static func parseFloat(flag: String, default defaultValue: Float) -> Float {
        let args = CommandLine.arguments
        guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return defaultValue }
        return Float(args[idx + 1]) ?? defaultValue
    }

    static func parseInt(flag: String, default defaultValue: Int) -> Int {
        let args = CommandLine.arguments
        guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return defaultValue }
        return Int(args[idx + 1]) ?? defaultValue
    }
}
