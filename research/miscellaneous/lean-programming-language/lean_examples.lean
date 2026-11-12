import Mathlib

/-!
Examples for “Introduction to Lean Programming Language”.
API references reviewed against the online Mathlib documentation on 2026-09-16.
Not locally compiled: Lean and Lake were unavailable on the authoring machine.
Run with `lake env lean lean_examples.lean` inside an existing Mathlib project.
The BEGIN/END markers delimit the exact examples embedded in the HTML page.
-/

namespace LeanProgrammingLanguage
open scoped BigOperators

-- BEGIN arithmetic
example : (2 : ℕ) + 3 = 5 := by
  norm_num

example (x : ℝ) : x = x := by
  rfl
-- END arithmetic

-- BEGIN algebra
theorem square_expansion (x y : ℝ) :
    (x + y)^2 = x^2 + 2*x*y + y^2 := by
  ring
-- END algebra

-- BEGIN logic
theorem swap_and (P Q : Prop) : P ∧ Q → Q ∧ P := by
  intro h
  constructor
  · exact h.2
  · exact h.1

example (P Q : Prop) : P ∧ Q → Q ∧ P :=
  fun h => ⟨h.2, h.1⟩
-- END logic

-- BEGIN rewrite
example (x y : ℝ) (h : x = y) : x + 1 = y + 1 := by
  rw [h]
-- END rewrite

-- BEGIN odd_sum
theorem sum_first_odds (n : ℕ) :
    (∑ k ∈ Finset.range n, (2*k + 1)) = n^2 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      ring
-- END odd_sum

-- BEGIN even_sum
theorem even_plus_even (m n : ℕ)
    (hm : ∃ a, m = 2*a) (hn : ∃ b, n = 2*b) :
    ∃ c, m + n = 2*c := by
  rcases hm with ⟨a, ha⟩
  rcases hn with ⟨b, hb⟩
  use a + b
  rw [ha, hb]
  ring
-- END even_sum

-- BEGIN linear
example (T : (Fin 2 → ℝ) →ₗ[ℝ] (Fin 2 → ℝ))
    (u v : Fin 2 → ℝ) : T (u + v) = T u + T v := by
  exact T.map_add u v

example (T : (Fin 2 → ℝ) →ₗ[ℝ] (Fin 2 → ℝ)) :
    T 0 = 0 := by
  exact T.map_zero

example (T : (Fin 2 → ℝ) →ₗ[ℝ] (Fin 2 → ℝ))
    (a b : ℝ) (u v : Fin 2 → ℝ) :
    T (a • u + b • v) = a • T u + b • T v := by
  simp only [map_add, map_smul]
-- END linear

-- BEGIN matrix
example (u v : Fin 2 → ℝ) :
    (![![2, 0], ![0, 1]] : Matrix (Fin 2) (Fin 2) ℝ).mulVec
      (u + v) =
    (![![2, 0], ![0, 1]] : Matrix (Fin 2) (Fin 2) ℝ).mulVec u +
    (![![2, 0], ![0, 1]] : Matrix (Fin 2) (Fin 2) ℝ).mulVec v := by
  exact Matrix.mulVec_add _ u v
-- END matrix

-- BEGIN square_deriv
theorem square_derivative (x : ℝ) :
    HasDerivAt (fun t : ℝ => t^2) (2*x) x := by
  simpa using (hasDerivAt_pow 2 x)
-- END square_deriv

-- BEGIN product_deriv
theorem square_times_exp (x : ℝ) :
    HasDerivAt (fun t : ℝ => t^2 * Real.exp t)
      (2*x * Real.exp x + x^2 * Real.exp x) x := by
  exact (square_derivative x).mul (Real.hasDerivAt_exp x)
-- END product_deriv

-- BEGIN affine
theorem affine_derivative
    (A : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ)) (b x : ℝ × ℝ) :
    HasFDerivAt (fun z => A z + b) A x := by
  exact A.hasFDerivAt.add_const b
-- END affine

-- BEGIN ode
theorem exponential_solution (a c t : ℝ) :
    HasDerivAt (fun s : ℝ => c * Real.exp (a*s))
      (a * (c * Real.exp (a*t))) t := by
  have h : HasDerivAt (fun s : ℝ => a*s) a t := by
    simpa using (hasDerivAt_id t).const_mul a
  simpa only [mul_comm, mul_left_comm, mul_assoc] using
    h.exp.const_mul c

example (a c : ℝ) : c * Real.exp (a*0) = c := by
  simp
-- END ode

-- BEGIN chain
theorem frechet_chain_rule
    {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (f : E → F) (g : F → G) (x : E)
    (A : E →L[ℝ] F) (B : F →L[ℝ] G)
    (hf : HasFDerivAt f A x)
    (hg : HasFDerivAt g B (f x)) :
    HasFDerivAt (g ∘ f) (B.comp A) x := by
  exact hg.comp x hf
-- END chain

-- BEGIN tactic_simp
example (x : ℝ) : x + 0 = x := by
  simp
-- END tactic_simp

-- BEGIN tactic_linarith
example (x : ℝ) (h : 2*x + 1 ≤ 5) : x ≤ 2 := by
  linarith
-- END tactic_linarith

-- BEGIN tactic_nlinarith
example (x : ℝ) : 0 ≤ x^2 + 1 := by
  nlinarith [sq_nonneg x]
-- END tactic_nlinarith

end LeanProgrammingLanguage
