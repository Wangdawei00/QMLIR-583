#ifndef QUANTUM_TRANSFORMS_CONSTANTFOLDING_TD
#define QUANTUM_TRANSFORMS_CONSTANTFOLDING_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/Quantum/QuantumOps.td"

def Zero : NativeCodeCall<"$_builder.getF64FloatAttr(0.0)">;
def Pi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI)">;
def NegPi : NativeCodeCall<"$_builder.getF64FloatAttr(-M_PI)">;
def HalfPi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI/2)">;
def NegHalfPi : NativeCodeCall<"$_builder.getF64FloatAttr(-M_PI/2)">;
def QuarterPi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI/4)">;
def NegQuarterPi : NativeCodeCall<"$_builder.getF64FloatAttr(-M_PI/4)">;

def : Pat<(PauliXGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Pi)),
             (ConstantOp (Zero)),
             (ConstantOp (Pi)),
             $q)>;

def : Pat<(PauliYGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Pi)),
             (ConstantOp (HalfPi)),
             (ConstantOp (HalfPi)),
             $q)>;

def : Pat<(PauliZGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (Pi)),
             $q)>;

def : Pat<(HadamardGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (HalfPi)),
             (ConstantOp (Zero)),
             (ConstantOp (Pi)),
             $q)>;

def : Pat<(PhaseGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (HalfPi)),
             $q)>;

def : Pat<(PhaseDaggerGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (NegHalfPi)),
             $q)>;

def : Pat<(TGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (QuarterPi)),
             $q)>;

def : Pat<(TDaggerGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (NegQuarterPi)),
             $q)>;

def : Pat<(RotateXOp $alpha, $q),
          (UniversalRotationGateOp
             $alpha,
             (ConstantOp (NegHalfPi)),
             (ConstantOp (HalfPi)),
             $q)>;

def : Pat<(RotateYOp $alpha, $q),
          (UniversalRotationGateOp
             $alpha,
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             $q)>;

def : Pat<(RotateZOp $alpha, $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             $alpha,
             $q)>;

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
             
#endif // QUANTUM_TRANSFORMS_CONVERT1QTOU_TD
