// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "metr-denoise",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/soniqo/speech-swift.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "metr-denoise",
            dependencies: [
                .product(name: "SpeechEnhancement", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ]
        ),
    ]
)
