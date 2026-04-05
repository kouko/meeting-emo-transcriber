import FluidAudio
import Foundation

@main
struct MetrDiarize {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            log("Usage: metr-diarize <audio.wav> [--threshold <float>] [--num-speakers <int>]")
            log("       metr-diarize --extract-embedding <audio.wav>")
            exit(1)
        }

        // Mode: extract single embedding (for enroll)
        if args.contains("--extract-embedding") {
            guard let idx = args.firstIndex(of: "--extract-embedding"),
                  idx + 1 < args.count else {
                log("Error: --extract-embedding requires an audio file path")
                exit(1)
            }
            await extractEmbedding(audioPath: args[idx + 1])
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

            // Build speaker embeddings (centroid per speaker)
            var speakerEmbeddings: [String: [Double]] = [:]
            if let db = result.speakerDatabase {
                for (speakerId, embedding) in db {
                    speakerEmbeddings[speakerId] = embedding.map { Double($0) }
                }
            }

            let json: [String: Any] = [
                "segments": segments,
                "speakers": speakerCount,
                "speaker_embeddings": speakerEmbeddings,
            ]
            let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        } catch {
            log("Error: \(error)")
            exit(1)
        }
    }

    // MARK: - Extract Embedding Mode

    static func extractEmbedding(audioPath: String) async {
        do {
            // Use diarization with num_speakers=1 to get a single speaker embedding
            let (diarizer, _) = try await loadDiarizer(threshold: 0.6, numSpeakers: 1)

            log("Extracting speaker embedding...")
            let url = URL(fileURLWithPath: audioPath)
            let result = try await diarizer.process(url)

            guard let db = result.speakerDatabase, let firstEntry = db.first else {
                log("Error: no embedding extracted")
                exit(1)
            }

            let embedding = firstEntry.value.map { Double($0) }
            log("Embedding extracted: \(embedding.count) dimensions")

            let json: [String: Any] = [
                "embedding": embedding,
                "dim": embedding.count,
                "model": "wespeaker_v2",
            ]
            let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
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
