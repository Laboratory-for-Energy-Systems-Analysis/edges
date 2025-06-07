"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import numpy as np

from .edgelcia import *
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse import diags, csr_matrix, coo_matrix
from highspy import Highs


class CostLCIA(EdgeLCIA):
    """
    Class that implements the calculation of the regionalized life cycle impact assessment (LCIA) results.
    Relies on bw2data.LCA class for inventory calculations and matrices.
    """

    def __init__(
        self,
        demand: dict,
        method: tuple,
        filepath: Optional[str] = None,
        parameters: Optional[dict] = None,
        weight: Optional[str] = "population",
        use_distributions: Optional[bool] = False,
        random_seed: Optional[int] = None,
        iterations: Optional[int] = 100,
    ):
        """
        Initialize a `CostLCIA` object for life cycle costing using the edges framework.

        This class inherits from `EdgeLCIA` but applies cost factors instead of environmental CFs.

        Parameters
        ----------
        cost_key : str, optional
            The key identifying the cost entry in the method file to be used (e.g., "USD_2020").
        default_cost : float, optional
            Default cost value to assign when a match is not found or price data is missing.
        *args, **kwargs :
            Passed through to the `EdgeLCIA` constructor (e.g., demand, method, parameters, etc.)

        Notes
        -----
        This class supports price-based LCC modeling using regionalized, symbolic, or defaulted cost data.
        """
        super().__init__(
            demand=demand,
            method=method,
            filepath=filepath,
            parameters=parameters,
            weight=weight,
            use_distributions=use_distributions,
            random_seed=random_seed,
            iterations=iterations,
        )
        self.technosphere_matrix_star = None
        self.price_vector = None
        self.logger.info(f"Initialized CostLCIA with method {self.method}")

    def lci(self) -> None:
        """
        Perform the life cycle inventory (LCI) phase for cost-based assessment.

        This overrides or extends the base `EdgeLCIA.lci()` to focus on price and cost-related flows.

        It identifies relevant technosphere and biosphere exchanges for which cost data will be applied.

        Must be called before `build_price_vector()` or `evaluate_cfs()`.
        """

        self.lca.lci()

        self.technosphere_flow_matrix = build_technosphere_edges_matrix(
            self.lca.technosphere_matrix, self.lca.supply_array, preserve_diagonal=True
        )
        self.technosphere_edges = set(
            list(zip(*self.technosphere_flow_matrix.nonzero()))
        )

        self.technosphere_flows = get_flow_matrix_positions(
            {k: v for k, v in self.lca.activity_dict.items()}
        )

        self.reversed_activity, _, self.reversed_biosphere = self.lca.reverse_dict()

        # Build technosphere flow lookups as in the original implementation.
        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }

    def build_price_vector(self):
        """
        Construct the vector of prices (or costs) aligned with the inventory exchanges.

        This vector maps each exchange (technosphere or biosphere) to a cost value, based on:
        - Explicit values in the method file
        - Symbolic expressions using parameters
        - Defaults when no match is found

        Behavior
        --------
        - Uses the `cost_key` specified in the method to extract cost values.
        - Supports symbolic cost expressions resolved via parameters or scenarios.
        - Stores result in the internal `characterization_matrix`.

        Notes
        -----
        - Must be called after `lci()` and `map_exchanges()`.
        - If cost distributions are defined, use `evaluate_cfs()` with `use_distributions=True`.
        """

        print("Build price vector")

        self.price_vector = np.zeros_like(self.lca.supply_array)

        if len(self.cfs_mapping) == 0:
            raise ValueError("No CFs found in the mapping. Cannot build price vector.")

        # Iterate over the CFs mapping to fill the price vector
        for cf in self.cfs_mapping:
            for position in cf.get("positions", []):
                # if we find a pair of positions along the diagonal
                if position[0] == position[1]:
                    # we take the value of the first one
                    self.price_vector[position[0]] = cf["value"]

        # print the number of zero values
        zero_count = np.count_nonzero(self.price_vector == 0)
        print(
            f"Number of zero values in price vector: {zero_count}, out of {len(self.price_vector)}"
        )

    def infer_missing_costs(self):
        """
        Fill in missing costs for exchanges that lack an assigned price.

        This method is useful for achieving full coverage in the cost matrix,
        especially when the method file is incomplete or only partially matches the inventory.

        Parameters
        ----------
        fallback_value : float
            The default cost to apply to unmatched exchanges.

        Notes
        -----
        - This complements `build_price_vector()` by ensuring no exchange is left unpriced.
        - Works only for technosphere flows.
        """

        print("Inferring missing costs in the technosphere matrix...")

        # --- Part 1: Normalize technosphere matrix to build A* ---
        T = self.lca.technosphere_matrix.tocsc()
        n = T.shape[0]
        diag = T.diagonal()

        # Replace small/zero diagonals to avoid instability
        min_diag_val = 1e-2
        capped_diag = np.where(diag < min_diag_val, min_diag_val, diag)
        inv_diag = 1.0 / capped_diag

        # Remove diagonal
        T_no_diag = T.copy()
        T_no_diag.setdiag(0)
        T_no_diag.eliminate_zeros()

        # Normalize and clip A_star values
        rows, cols = T_no_diag.nonzero()
        raw_data = -T_no_diag.data * inv_diag[cols]
        max_abs_A = 1e3
        clipped_data = np.clip(raw_data, -max_abs_A, max_abs_A)

        A_star = csr_matrix((clipped_data, (rows, cols)), shape=T.shape)
        self.technosphere_matrix_star = A_star

        # --- Part 2: Build LP system ---
        original_prices = self.price_vector.copy()
        A_ub_rows = []
        b_ub = []
        bounds = []

        T_csc = T  # already in CSC format

        for j in range(n):
            col = T_csc[:, j]
            is_independent = col.nnz == 1 and col[j, 0] != 0
            price = original_prices[j]

            if is_independent:
                bounds.append((price, price))
                continue

            # Constraint: p_j ≥ sum(A*_ij * p_i)
            row = A_star[:, j].tocoo()
            coeffs = np.zeros(n)
            coeffs[row.row] = row.data
            coeffs[j] -= 1  # move p_j to LHS

            A_ub_rows.append(coeffs)
            b_ub.append(0)

            bounds.append((price, None) if price > 0 else (0, None))

        if not A_ub_rows:
            raise RuntimeError("No constraints generated; check matrix content.")

        A_ub = np.vstack(A_ub_rows)
        b_ub = np.array(b_ub)
        c = np.ones(n)

        # --- Diagnostics ---
        print("Price vector stats:")
        print("  Min:", np.min(original_prices))
        print("  Max:", np.max(original_prices))
        print("  NaNs:", np.isnan(original_prices).sum())
        print("  Zeros:", np.count_nonzero(original_prices == 0))
        print(
            "  Anchored prices:",
            sum(lb == ub and lb is not None for lb, ub in bounds),
            "/",
            n,
        )

        print("A_ub diagnostics:")
        print("  Max abs value:", np.max(np.abs(A_ub)))
        print(
            "  Non-zeros per row (min/max):",
            np.min(np.count_nonzero(A_ub, axis=1)),
            "/",
            np.max(np.count_nonzero(A_ub, axis=1)),
        )
        print("  Total constraints (rows):", A_ub.shape[0])
        print("  Total variables (columns):", A_ub.shape[1])

        assert np.all(np.isfinite(A_ub)), "A_ub contains non-finite values"
        assert np.all(np.isfinite(b_ub)), "b_ub contains non-finite values"
        for lb, ub in bounds:
            assert lb is None or np.isfinite(lb), f"Bad lower bound: {lb}"
            assert ub is None or np.isfinite(ub), f"Bad upper bound: {ub}"

        # --- Solve LP ---
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if result.success:
            zero_count_before = np.count_nonzero(self.price_vector == 0)
            zero_count_after = np.count_nonzero(result.x == 0)
            self.price_vector = result.x
            print(
                f"Inferred {zero_count_before - zero_count_after} missing prices successfully. "
                f"Remaining {zero_count_after} missing prices."
            )
        else:
            print("Linear programming failed.")
            print("Status:", result.status)
            print("Message:", result.message)
            print("Number of variables:", len(c))
            print("Number of constraints:", len(b_ub))
            raise RuntimeError(
                "Linear program failed to find a consistent price vector."
            )

    def infer_missing_costs_highspy(self):
        """
        Infer missing costs using the HighSPy logic for structured economic data.

        This method is specific to applications where costs can be inferred from structured
        product classifications (e.g., CPC, HS) or external statistical sources.

        Behavior
        --------
        - Applies heuristics or classification-based rules to assign prices when direct matches are absent.
        - Typically used in conjunction with structured trade/economic datasets (e.g., BACI, World Bank, etc.)

        Notes
        -----
        - This method assumes the inventory and/or CFs contain classifications that can be used for inference.
        - Supports a fallback pricing strategy that complements `build_price_vector()` and `infer_missing_costs()`.
        """

        print("Inferring missing costs using HiGHS...")

        T = self.lca.technosphere_matrix.tocsc()
        n = T.shape[0]
        diag = T.diagonal()
        valid_cols = np.flatnonzero(diag)
        inv_diag = np.zeros_like(diag)
        inv_diag[valid_cols] = 1.0 / diag[valid_cols]

        T_no_diag = T.copy()
        T_no_diag.setdiag(0)
        T_no_diag.eliminate_zeros()

        rows, cols = T_no_diag.nonzero()
        data = -T_no_diag.data * inv_diag[cols]
        A_star = csr_matrix((data, (rows, cols)), shape=T.shape)

        bounds = []
        A_ub_rows = []
        A_ub_cols = []
        A_ub_data = []
        b_ub = []

        for j in range(n):
            col = T[:, j]
            is_independent = col.nnz == 1 and col[j, 0] != 0
            price = self.price_vector[j]

            if is_independent:
                bounds.append((price, price))
                continue

            col_A_star = A_star[:, j].tocoo()
            for i, v in zip(col_A_star.row, col_A_star.data):
                A_ub_rows.append(len(b_ub))
                A_ub_cols.append(i)
                A_ub_data.append(v)
            A_ub_rows.append(len(b_ub))
            A_ub_cols.append(j)
            A_ub_data.append(-1.0)

            b_ub.append(0.0)
            bounds.append((price, None) if price > 0 else (0, None))

        model = Highs()
        model.setOptionValue("output_flag", False)

        model.passModel(
            {
                "num_col": n,
                "num_row": len(b_ub),
                "sense": "min",
                "col_cost": np.ones(n),
                "col_bounds": bounds,
                "row_bounds": [(None, b) for b in b_ub],
                "a_matrix": {
                    "format": "coo",
                    "start": None,
                    "index": list(zip(A_ub_rows, A_ub_cols)),
                    "value": A_ub_data,
                },
            }
        )

        model.run()
        status = model.getModelStatus()

        if status == model.ModelStatus.OPTIMAL:
            solution = model.getSolution().col_value
            print("Inference successful.")
            return np.array(solution)
        else:
            raise RuntimeError(f"HiGHS failed: status {status}")

    def evaluate_cfs(self, scenario_idx: str | int = 0, scenario=None):
        """
        Evaluate and apply cost values for each matched exchange.

        This method overrides or extends the standard `EdgeLCIA.evaluate_cfs()` to work with cost values.
        It supports:
        - Direct cost values (floats)
        - Symbolic expressions involving scenario-dependent parameters
        - Distributions for uncertainty propagation

        Parameters
        ----------
        scenario_idx : str or int, optional
            The time or scenario index (e.g., year or version), used to select parameter values.
        scenario : str, optional
            Name of the scenario to evaluate (overrides the default from init).

        Notes
        -----
        - The resulting values populate the `characterization_matrix`, used in cost impact calculations.
        - Can be used to build scenario-based or dynamic LCC models.
        """

        if self.use_distributions and self.iterations > 1:
            coords_i, coords_j, coords_k = [], [], []
            data = []

            for cf in self.cfs_mapping:
                samples = sample_cf_distribution(
                    cf=cf,
                    n=self.iterations,
                    parameters=self.parameters,
                    random_state=self.random_state,
                    use_distributions=self.use_distributions,
                    SAFE_GLOBALS=self.SAFE_GLOBALS,
                )
                for i, j in cf["positions"]:
                    for k in range(self.iterations):
                        coords_i.append(i)
                        coords_j.append(j)
                        coords_k.append(k)
                        data.append(samples[k])

            n_rows, n_cols = self.lca.technosphere_matrix.shape

            self.characterization_matrix = sparse.COO(
                coords=[coords_i, coords_j, coords_k],
                data=data,
                shape=(n_rows, n_cols, self.iterations),
            )

            self.scenario_cfs = [{"positions": [], "value": 0}]  # dummy

        else:
            # Fallback to 2D
            self.scenario_cfs = []
            scenario_name = None

            if scenario is not None:
                scenario_name = scenario
            elif self.scenario is not None:
                scenario_name = self.scenario

            if scenario_name is None:
                if isinstance(self.parameters, dict):
                    if len(self.parameters) > 0:
                        scenario_name = list(self.parameters.keys())[0]

            resolved_params = self._resolve_parameters_for_scenario(
                scenario_idx, scenario_name
            )

            for cf in self.cfs_mapping:
                if isinstance(cf["value"], str):

                    value = safe_eval_cached(
                        cf["value"],
                        parameters=resolved_params,
                        scenario_idx=scenario_idx,
                        SAFE_GLOBALS=self.SAFE_GLOBALS,
                    )
                else:
                    value = cf["value"]

                self.scenario_cfs.append(
                    {
                        "supplier": cf["supplier"],
                        "consumer": cf["consumer"],
                        "positions": cf["positions"],
                        "value": value,
                    }
                )

            matrix_type = "technosphere"

            self.characterization_matrix = initialize_lcia_matrix(
                self.lca, matrix_type=matrix_type
            )

            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    self.characterization_matrix[i, j] = cf["value"]

            self.characterization_matrix = self.characterization_matrix.tocsr()

            self._post_process_characterization_matrix()

    def _post_process_characterization_matrix(self):
        """
        Post-process the characterization matrix to ensure that:
        1. The diagonal is overwritten with the price vector.
        2. The off-diagonal technosphere flow values are flipped.
        3. The price vector is added to the characterization matrix
              wherever the technosphere flow matrix is non-zero.
        4. The characterization matrix is converted to CSR format.

        This method is called after the characterization matrix has been
        generated and the price vector has been built.
        It modifies the characterization matrix in place.

        """

        # Step 1: Overwrite diagonal of characterization matrix with price vector
        n = self.characterization_matrix.shape[0]
        new_diag = diags(self.price_vector, offsets=0, shape=(n, n), format="csr")
        self.characterization_matrix.setdiag(0)
        self.characterization_matrix = self.characterization_matrix + new_diag

        # Step 2: Flip signs of off-diagonal technosphere flow values
        A_coo = self.technosphere_flow_matrix.tocoo()
        rows, cols, data = A_coo.row, A_coo.col, A_coo.data.copy()
        data[rows != cols] *= -1
        self.technosphere_flow_matrix = csr_matrix(
            (data, (rows, cols)), shape=A_coo.shape
        )

        # Step 3: Add price_vector[i] (not j!) to characterization_matrix[i, j]
        # Convert technosphere flow matrix to COO format for indexing
        T = self.technosphere_flow_matrix.tocoo()
        C = self.characterization_matrix.tocsr()

        # Determine where characterization_matrix is zero
        # This builds a mask for locations (i,j) where:
        # - there's a flow
        # - but no existing characterization value
        is_zero_in_C = C[T.row, T.col].A1 == 0  # .A1 flattens to 1D
        # Filter only the (i,j) where C is currently zero
        rows = T.row[is_zero_in_C]
        cols = T.col[is_zero_in_C]
        data = self.price_vector[rows]  # Add row-wise prices

        # Build sparse delta matrix with new values
        delta_matrix = coo_matrix((data, (rows, cols)), shape=C.shape)

        # Update the characterization matrix (add only to empty entries)
        self.characterization_matrix = (C + delta_matrix).tocsr()

    def generate_cf_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with the evaluated characterization factors
        used in the current scenario. If distributions were used, show summary statistics.
        """

        if not self.scenario_cfs:
            print("You must run evaluate_cfs() first.")
            return pd.DataFrame()

        is_biosphere = False
        inventory = self.technosphere_flow_matrix

        data = []

        # Deterministic fallback
        for cf in self.scenario_cfs:
            for i, j in cf["positions"]:
                consumer = bw2data.get_activity(self.reversed_activity[j])
                supplier = bw2data.get_activity(self.reversed_activity[i])

                amount = inventory[i, j]
                cf_value = cf["value"]
                impact = amount * cf_value

                cpc_supplier, cpc_consumer = None, None
                if "classifications" in supplier:
                    for classification in supplier["classifications"]:
                        if classification[0].lower() == "cpc":
                            cpc_supplier = classification[1].split(":")[0].strip()
                            break

                if "classifications" in consumer:
                    for classification in consumer["classifications"]:
                        if classification[0].lower() == "cpc":
                            cpc_consumer = classification[1].split(":")[0].strip()
                            break

                entry = {
                    "supplier name": supplier["name"],
                    "supplier reference product": supplier.get("reference product"),
                    "supplier location": supplier.get("location"),
                    "supplier cpc": cpc_supplier,
                    "consumer name": consumer["name"],
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "consumer cpc": cpc_consumer,
                    "amount": amount,
                    "CF": cf_value,
                    "impact": impact,
                }
                data.append(entry)

        # we add unprocessed_edges
        for i, j in self.unprocessed_technosphere_edges:
            consumer = bw2data.get_activity(self.reversed_activity[j])
            supplier = bw2data.get_activity(self.reversed_activity[i])

            amount = inventory[i, j]
            cf_value = None
            impact = None

            cpc_supplier, cpc_consumer = None, None
            if "classifications" in supplier:
                for classification in supplier["classifications"]:
                    if classification[0].lower() == "cpc":
                        cpc_supplier = classification[1].split(":")[0].strip()
                        break

            if "classifications" in consumer:
                for classification in consumer["classifications"]:
                    if classification[0].lower() == "cpc":
                        cpc_consumer = classification[1].split(":")[0].strip()
                        break

            entry = {
                "supplier name": supplier["name"],
                "supplier reference product": supplier.get("reference product"),
                "supplier location": supplier.get("location"),
                "supplier cpc": cpc_supplier,
                "consumer name": consumer["name"],
                "consumer reference product": consumer.get("reference product"),
                "consumer location": consumer.get("location"),
                "consumer cpc": cpc_consumer,
                "amount": amount,
                "CF": cf_value,
                "impact": impact,
            }
            data.append(entry)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Order columns
        preferred_columns = [
            "supplier name",
            "supplier reference product",
            "supplier location",
            "supplier cpc",
            "consumer name",
            "consumer reference product",
            "consumer location",
            "consumer cpc",
            "amount",
            "CF",
            "impact",
        ]

        df = df[[col for col in preferred_columns if col in df.columns]]

        return df
