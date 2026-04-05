import FluidAudio
import Foundation

@main
struct MetrDiarize {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            log("Usage: metr-diarize <audio.wav> [--threshold <float>] [--num-speakers <int>]")
            exit(1)
        }

        let audioPath = args[1]
        let threshold = parseFloat(flag: "--threshold", default: 0.6)
        let numSpeakers = parseInt(flag: "--num-speakers", default: 0)

        do {
            // 1. Load models (auto-download on first run)
            log("Loading diarization models...")
            let models = try await OfflineDiarizerModels.load { progress in
                log("  Downloading models: \(String(format: "%.0f", progress.fractionCompleted * 100))%")
            }

            // 2. Configure diarizer
            var config = OfflineDiarizerConfig.default
            config.clustering.threshold = Double(threshold)
            if numSpeakers > 0 {
                config = config.withSpeakers(exactly: numSpeakers)
            }

            let diarizer = OfflineDiarizerManager(config: config)
            diarizer.initialize(models: models)

            // 3. Process audio file
            log("Running diarization...")
            let url = URL(fileURLWithPath: audioPath)
            let result = try await diarizer.process(url)

            log("Diarization complete: \(result.segments.count) segments, \(Set(result.segments.map { $0.speakerId }).count) speakers")

            // 4. Output JSON to stdout
            let output = result.segments.map { seg -> [String: Any] in
                [
                    "start": Double(seg.startTimeSeconds),
                    "end": Double(seg.endTimeSeconds),
                    "speaker": seg.speakerId,
                ]
            }
            let json: [String: Any] = [
                "segments": output,
                "speakers": Set(result.segments.map { $0.speakerId }).count,
            ]
            let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        } catch {
            log("Error: \(error)")
            exit(1)
        }
    }

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
