"""Markdown optimisation with PuLP under fairness constraints."""
import pulp


def recommend_markdowns(sku_forecasts: dict, holding_cost: float, max_discount: float = 0.5) -> dict:
    """Return the optimal markdown per SKU given forecasts and holding cost.

    This is a stub showing the solver pattern. The Sprint 3 ticket in the
    Module 2 plan will expand it with revenue uplift curves and the Module 1
    fairness constraints (no region may receive markdowns outside the defined
    band from the median).
    """
    model = pulp.LpProblem("markdown_optimisation", pulp.LpMinimize)
    discounts = {sku: pulp.LpVariable(f"d_{sku}", 0, max_discount) for sku in sku_forecasts}
    model += pulp.lpSum(discounts[s] * holding_cost for s in sku_forecasts)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return {sku: v.value() for sku, v in discounts.items()}
