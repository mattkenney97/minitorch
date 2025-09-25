from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals_forward = list(vals)
    vals_backward = list(vals)
    vals_forward[arg] = vals_forward[arg] + (epsilon / 2)
    vals_backward[arg] = vals_forward[arg] - (epsilon / 2)

    return (f(*vals_forward) - f(*vals_backward)) / (epsilon / 2)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    seen = set()
    result = []

    def visit(node: Variable, res: list):
        if node.name in seen:
            return res

        # if not node.is_constant:
        if node.history is not None:
            for parent in node.parents:
                res = visit(parent, res)
        res = [node] + res
        seen.add(node.name)
        return res

    result = visit(variable, result)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    scalars = topological_sort(variable)

    deriv_dict = {variable.name: deriv}
    for scalar in scalars:
        if scalar.is_leaf():
            scalar.accumulate_derivative(deriv_dict[scalar.name])
        elif not scalar.is_constant():
            derivs = scalar.chain_rule(deriv_dict[scalar.name])
            for d in derivs:
                if d[0].name in deriv_dict:
                    deriv_dict[d[0].name] += d[1]
                else:
                    deriv_dict[d[0].name] = d[1]


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
