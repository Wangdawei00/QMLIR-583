add_mlir_dialect_library(QuantumTransforms
    QuantumRewrites.cpp
    Convert1QToU.cpp
    PrepareForZX.cpp
    Toy.cpp
    ConstantFolding.cpp
    DeadCodeElimination.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Quantum/Transforms

    DEPENDS
    MLIRQuantumPassIncGen
    MLIRQuantumTransformsRewritesPassIncGen
    MLIRQuantumConvert1QToUPassIncGen
    MLIRQuantumPrepareForZXPassIncGen
    MLIRZXTransformsRewritesPassIncGen
    MLIRConstantFoldingPassIncGen
    )

target_link_libraries(QuantumTransforms PUBLIC MLIRIR)
