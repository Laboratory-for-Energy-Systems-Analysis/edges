
Quick Start
===========

This minimal example walks you through performing a basic LCIA using `edges`.

.. code-block:: python

    import bw2data
    from edges import EdgeLCIA, get_available_methods

    bw2data.project.set_current("some project")
    act = bw2data.Database("some db").random()

    # get available method
    get_available_methods()

    # Load a built-in method
    lcia = EdgeLCIA(
        demand={act: 1},
        method=("AWARE 2.0", "Country", "all", "yearly")
    )

    # Step 1: Build the inventory
    lcia.lci()

    # Step 2.a: Match exchanges to characterization factors
    lcia.map_exchanges()
    # Step 2.b: since this si a regionalized method, a few more steps are required
    lcia.map_aggregate_locations() # finds matches for aggregate regions ("RER", "US" etc.)
    lcia.map_dynamic_locations() # finds matches for dynamic regions ("RoW", "RoW", etc.)
    lcia.map_contained_locations() # finds matches for contained regions ("CA" for "CA-QC" if factor of "CA-QC" is not available)
    lcia.map_remaining_locations_to_global() # applies global factors to remaining locations

    # Step 3: Evaluate CFs (e.g., resolve symbolic expressions)
    lcia.evaluate_cfs()

    # Step 4: Compute the LCIA score
    lcia.lcia()

    # Step 5 (optional): Print a summary
    print(lcia.statistics())

    # Step 6 (optional but RECOMMENDED): Print a table with all exchanges characterized
    # this allows you to check whether exchanges have been given the correct CFs
    # include_unmatched=True allows you to see which exchanges were not matched (and if some should have been)
    df = lcia.generate_cf_table(include_unmatched=False)
