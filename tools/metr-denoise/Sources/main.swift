import AudioCommon
import Foundation
import SpeechEnhancement

@main
struct MetrDenoise {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 3 else {
            log("Usage: metr-denoise <input.wav> <output.wav>")
            log("  Enhance speech audio using DeepFilterNet3 (MLX Metal GPU)")
            exit(1)
        }

        let inputPath = args[1]
        let outputPath = args[2]

        do {
            // 1. Load model (auto-download on first run)
            log("Loading DeepFilterNet3 model...")
            let enhancer = try await SpeechEnhancer.fromPretrained(
                progressHandler: { @Sendable progress, status in
                    MetrDenoise.log("  \(status) \(Int(progress * 100))%")
                }
            )

            // 2. Load input audio
            log("Loading audio: \(inputPath)")
            let inputURL = URL(fileURLWithPath: inputPath)
            let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: inputURL)
            let durationSec = Float(samples.count) / Float(sampleRate)
            log("  Duration: \(String(format: "%.1f", durationSec))s, sample rate: \(sampleRate) Hz")

            // 3. Enhance
            log("Enhancing audio...")
            let startTime = Date()
            let cleanSamples = try enhancer.enhance(audio: samples, sampleRate: sampleRate)
            let elapsed = Date().timeIntervalSince(startTime)
            let rtf = elapsed / Double(durationSec)
            log("  Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")

            // 4. Save output (DeepFilterNet3 outputs 48kHz)
            let outputURL = URL(fileURLWithPath: outputPath)
            try WAVWriter.write(samples: cleanSamples, sampleRate: SpeechEnhancer.sampleRate, to: outputURL)
            log("Written: \(outputPath)")

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
}
