// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "metr-diarize",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "metr-diarize",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ]
        ),
    ]
)
