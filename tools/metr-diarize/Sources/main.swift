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

            log("Running diarization...")
            let url = URL(fileURLWithPath: audioPath)
            let result = try await diarizer.process(url)

            let speakerCount = Set(result.segments.map { $0.speakerId }).count
            log("Diarization complete: \(result.segments.count) segments, \(speakerCount) speakers")

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
