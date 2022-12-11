#ifndef QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
#define QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/Quantum/QuantumOps.td"
// def : Pat<(CNOTGateOp
//             (CNOTGateOp $q0, $q1):$res0__0)
//             (CNOTGateOp $q0, $q1):$res1__1)),
//           [(replaceWithValue $q0), (replaceWithValue $q1)]>;

// S.S = Z
def : Pat<(PhaseGateOp (PhaseGateOp $q)), (PauliZGateOp $q)>;
// S^-1 . S^-1 = Z
def : Pat<(PhaseDaggerGateOp (PhaseDaggerGateOp $q)), (PauliZGateOp $q)>;
// T.T = S
def : Pat<(TGateOp (TGateOp $q)), (PhaseGateOp $q)>;
// T^-1 . T^-1 = S^-1
def : Pat<(TDaggerGateOp (TDaggerGateOp $q)), (PhaseDaggerGateOp $q)>;

def : Pat<(RotateXOp $beta, (RotateXOp $alpha, $q)), (RotateXOp (AddFOp $alpha, $beta), $q)>;
def : Pat<(RotateYOp $beta, (RotateYOp $alpha, $q)), (RotateYOp (AddFOp $alpha, $beta), $q)>;
def : Pat<(RotateZOp $beta, (RotateZOp $alpha, $q)), (RotateZOp (AddFOp $alpha, $beta), $q)>;

// Merge adjacent U ops, if all angles are constants
def ZYZa1 : NativeCodeCall<"EulerAngles::wrapZYZa1($_builder, $0, $1, $2, $3)">;
def ZYZa2 : NativeCodeCall<"EulerAngles::wrapZYZa2($_builder, $0, $1, $2, $3)">;
def ZYZa3 : NativeCodeCall<"EulerAngles::wrapZYZa3($_builder, $0, $1, $2, $3)">;
// U(...).U(...) = U(...)
// U(theta, phi, lambda)
//           = U(theta2, phi2, lambda2).U(theta1, phi1, lambda1)
//           = Rz(lambda2).Ry(theta2).Rz(phi2+lambda1).Ry(theta1).Rz(phi1)
//           = Rz(lambda2).Rz(lambda').Ry(theta').Rz(phi').Rz(phi1)
//           = U(theta', phi1 + phi', lambda2 + lambda')
def : Pat<(UniversalRotationGateOp
            (ConstantOp $theta2),
            (ConstantOp $phi2),
            $lambda2,
            (UniversalRotationGateOp
               (ConstantOp $theta1),
               $phi1,
               (ConstantOp $lambda1),
               $q)),
          (UniversalRotationGateOp
             (ConstantOp (ZYZa2 $theta1, $phi2, $lambda1, $theta2)),
             (AddFOp $phi1, (ConstantOp (ZYZa1 $theta1, $phi2, $lambda1, $theta2))),
             (AddFOp $lambda2, (ConstantOp (ZYZa3 $theta1, $phi2, $lambda1, $theta2))),
             $q)>;

// https://qiskit.org/documentation/stubs/qiskit.transpiler.passes.RemoveDiagonalGatesBeforeMeasure.html
def : Pat<(Quantum_MeasureOp (RotateZOp $param, $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (PhaseGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (PhaseDaggerGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (TGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (TDaggerGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (PauliZGateOp $q)), (Quantum_MeasureOp $q)>;

def : Pat<(Quantum_MeasureQubitOp (RotateZOp $param, $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (PhaseGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (PhaseDaggerGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (TGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (TDaggerGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (PauliZGateOp $q)), (Quantum_MeasureQubitOp $q)>;

#endif // QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
