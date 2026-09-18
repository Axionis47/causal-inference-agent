"""The web server: one FastAPI app that lists and writes datasets, drives the desk graph, and serves run artifacts.

It adds no model calls and no analysis logic. Facts on disk, the graph's own judgements, and a projection of its
state for the page."""
