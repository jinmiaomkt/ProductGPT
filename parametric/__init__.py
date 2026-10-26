"""R46: parametric market-response benchmarks on ProductGPT's own rows.

    data.py      export the occasion table: the neural models' rows, window
                 and customer partition, reproduced from build_loaders
    state.py     the observed pre-decision state every rung reads
    model.py     the rung ladder R0-R6 as one functional choice model
    fit.py       estimate one rung, score the four cells, write results
    compare.py   paired customer-bootstrap differences between rungs
    leakcheck.py G2: every state variable against the label
    simulate.py  forward simulation from a fitted rung (G3 recovery, part B)

Everything here runs on CPU. On HPCC it runs on the short/long queues as
parallel array jobs (scripts/r46_*.pbs), never on the GPU queue.
"""
