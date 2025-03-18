//! # RAP (Randomized Air with Preprocessing)
//! See <https://hackmd.io/@aztec-network/plonk-arithmetiization-air> for formal definition.

use core::ops::{Add, Mul, Sub};
use std::{
    any::{type_name, Any},
    sync::Arc,
};

use p3_field::{ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    air_builders::{debug::DebugConstraintBuilder, symbolic::SymbolicRapBuilder},
    config::{StarkGenericConfig, Val},
};

/// An AIR (algebraic intermediate representation).
pub trait BaseAir<F>: Sync {
    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    fn columns(&self) -> Vec<String>;
}

/// An AIR with 0 or more public values.
/// This trait will be merged into Plonky3 in PR: <https://github.com/Plonky3/Plonky3/pull/470>
pub trait BaseAirWithPublicValues<F>: BaseAir<F> {
    fn num_public_values(&self) -> usize {
        0
    }

    fn columns(&self) -> Vec<String>;
}

/// An AIR with 1 or more main trace partitions.
pub trait PartitionedBaseAir<F>: BaseAir<F> {
    /// By default, an AIR has no cached main trace.
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![]
    }
    /// By default, an AIR has only one private main trace.
    fn common_main_width(&self) -> usize {
        self.width()
    }
}

/// An AIR that works with a particular `AirBuilder`.
pub trait Air<AB: AirBuilder>: BaseAir<AB::F> {
    fn eval(&self, builder: &mut AB);
}

pub trait AirBuilder: Sized {
    type F: Field;

    type Expr: FieldAlgebra
        + From<Self::F>
        + Add<Self::Var, Output = Self::Expr>
        + Add<Self::F, Output = Self::Expr>
        + Sub<Self::Var, Output = Self::Expr>
        + Sub<Self::F, Output = Self::Expr>
        + Mul<Self::Var, Output = Self::Expr>
        + Mul<Self::F, Output = Self::Expr>;

    type Var: Into<Self::Expr>
        + Copy
        + Send
        + Sync
        + Add<Self::F, Output = Self::Expr>
        + Add<Self::Var, Output = Self::Expr>
        + Add<Self::Expr, Output = Self::Expr>
        + Sub<Self::F, Output = Self::Expr>
        + Sub<Self::Var, Output = Self::Expr>
        + Sub<Self::Expr, Output = Self::Expr>
        + Mul<Self::F, Output = Self::Expr>
        + Mul<Self::Var, Output = Self::Expr>
        + Mul<Self::Expr, Output = Self::Expr>;

    type M: Matrix<Self::Var>;

    fn main(&self) -> Self::M;

    fn is_first_row(&self) -> Self::Expr;
    fn is_last_row(&self) -> Self::Expr;
    fn is_transition(&self) -> Self::Expr {
        self.is_transition_window(2)
    }
    fn is_transition_window(&self, size: usize) -> Self::Expr;

    /// Returns a sub-builder whose constraints are enforced only when `condition` is nonzero.
    fn when<I: Into<Self::Expr>>(&mut self, condition: I) -> FilteredAirBuilder<'_, Self> {
        FilteredAirBuilder {
            inner: self,
            condition: condition.into(),
        }
    }

    /// Returns a sub-builder whose constraints are enforced only when `x != y`.
    fn when_ne<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(
        &mut self,
        x: I1,
        y: I2,
    ) -> FilteredAirBuilder<'_, Self> {
        self.when(x.into() - y.into())
    }

    /// Returns a sub-builder whose constraints are enforced only on the first row.
    fn when_first_row(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_first_row())
    }

    /// Returns a sub-builder whose constraints are enforced only on the last row.
    fn when_last_row(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_last_row())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last.
    fn when_transition(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_transition())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last `size - 1`.
    fn when_transition_window(&mut self, size: usize) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_transition_window(size))
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I);

    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::Expr::ONE);
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean, i.e. either 0 or 1.
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.assert_zero(x.clone() * (x - Self::Expr::ONE));
    }

    /// Assert that `x` is ternary, i.e. either 0, 1 or 2.
    fn assert_tern<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.assert_zero(x.clone() * (x.clone() - Self::Expr::ONE) * (x - Self::Expr::TWO));
    }
}

pub trait AirBuilderWithPublicValues: AirBuilder {
    type PublicVar: Into<Self::Expr> + Copy;

    fn public_values(&self) -> &[Self::PublicVar];
}

pub trait PairBuilder: AirBuilder {
    fn preprocessed(&self) -> Self::M;
}

pub trait ExtensionBuilder: AirBuilder {
    type EF: ExtensionField<Self::F>;

    type ExprEF: FieldExtensionAlgebra<Self::Expr, F = Self::EF>;

    type VarEF: Into<Self::ExprEF> + Copy + Send + Sync;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>;

    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x.into() - y.into());
    }

    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext(x, Self::ExprEF::ONE)
    }
}

pub trait PermutationAirBuilder: ExtensionBuilder {
    type MP: Matrix<Self::VarEF>;

    type RandomVar: Into<Self::ExprEF> + Copy;

    fn permutation(&self) -> Self::MP;

    fn permutation_randomness(&self) -> &[Self::RandomVar];
}

#[derive(Debug)]
pub struct FilteredAirBuilder<'a, AB: AirBuilder> {
    pub inner: &'a mut AB,
    condition: AB::Expr,
}

impl<AB: AirBuilder> FilteredAirBuilder<'_, AB> {
    pub fn condition(&self) -> AB::Expr {
        self.condition.clone()
    }
}

impl<AB: AirBuilder> AirBuilder for FilteredAirBuilder<'_, AB> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = AB::M;

    fn main(&self) -> Self::M {
        self.inner.main()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(self.condition() * x.into());
    }
}

impl<AB: ExtensionBuilder> ExtensionBuilder for FilteredAirBuilder<'_, AB> {
    type EF = AB::EF;
    type ExprEF = AB::ExprEF;
    type VarEF = AB::VarEF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.inner.assert_zero_ext(x.into() * self.condition());
    }
}

impl<AB: PermutationAirBuilder> PermutationAirBuilder for FilteredAirBuilder<'_, AB> {
    type MP = AB::MP;

    type RandomVar = AB::RandomVar;

    fn permutation(&self) -> Self::MP {
        self.inner.permutation()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.inner.permutation_randomness()
    }
}

/// An AIR that works with a particular `AirBuilder` which allows preprocessing
/// and injected randomness.
///
/// Currently this is not a fully general RAP. Only the following phases are allowed:
/// - Preprocessing
/// - Main trace generation and commitment
/// - Permutation trace generation and commitment
///
/// Randomness is drawn after the main trace commitment phase, and used in the permutation trace.
///
/// Does not inherit [Air](p3_air::Air) trait to allow overrides for technical reasons
/// around dynamic dispatch.
pub trait Rap<AB>: Sync
where
    AB: PermutationAirBuilder,
{
    fn eval(&self, builder: &mut AB);
}

/// Permutation AIR builder that exposes certain values to both prover and verifier
/// _after_ the permutation challenges are drawn. These can be thought of as
/// "public values" known after the challenges are drawn.
///
/// Exposed values are used internally by the prover and verifier
/// in cross-table permutation arguments.
pub trait PermutationAirBuilderWithExposedValues: PermutationAirBuilder {
    fn permutation_exposed_values(&self) -> &[Self::VarEF];
}

/// Shared reference to any Interactive Air.
/// This type is the main interface for keygen.
pub type AirRef<SC> = Arc<dyn AnyRap<SC>>;

/// RAP trait for all-purpose dynamic dispatch use.
/// This trait is auto-implemented if you implement `Air` and `BaseAirWithPublicValues` and `PartitionedBaseAir` traits.
pub trait AnyRap<SC: StarkGenericConfig>:
Rap<SymbolicRapBuilder<Val<SC>>> // for keygen to extract fixed data about the RAP
    + for<'a> Rap<DebugConstraintBuilder<'a, SC>> // for debugging
    + BaseAirWithPublicValues<Val<SC>>
    + PartitionedBaseAir<Val<SC>>
    + Send + Sync
{
    fn as_any(&self) -> &dyn Any;
    /// Name for display purposes
    fn name(&self) -> String;
}

impl<SC, T> AnyRap<SC> for T
where
    SC: StarkGenericConfig,
    T: Rap<SymbolicRapBuilder<Val<SC>>>
        + for<'a> Rap<DebugConstraintBuilder<'a, SC>>
        + BaseAirWithPublicValues<Val<SC>>
        + PartitionedBaseAir<Val<SC>>
        + Send
        + Sync
        + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> String {
        get_air_name(self)
    }
}

/// Automatically derives the AIR name from the type name for pretty display purposes.
pub fn get_air_name<T>(_rap: &T) -> String {
    let full_name = type_name::<T>().to_string();
    // Split the input by the first '<' to separate the main type from its generics
    if let Some((main_part, generics_part)) = full_name.split_once('<') {
        // Extract the last segment of the main type
        let main_type = main_part.split("::").last().unwrap_or("");

        // Remove the trailing '>' from the generics part and split by ", " to handle multiple generics
        let generics: Vec<String> = generics_part
            .trim_end_matches('>')
            .split(", ")
            .map(|generic| {
                // For each generic type, extract the last segment after "::"
                generic.split("::").last().unwrap_or("").to_string()
            })
            .collect();

        // Join the simplified generics back together with ", " and format the result
        format!("{}<{}>", main_type, generics.join(", "))
    } else {
        // If there's no generic part, just return the last segment after "::"
        full_name.split("::").last().unwrap_or("").to_string()
    }
}
