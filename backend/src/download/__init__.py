"""Download subsystem.

Owns Kaggle credentials, dataset acquisition, and on-disk dataset storage.
Analysis stages consume DownloadRecord from this module; they do not call
Kaggle directly.

The public surface is filled in at the end of the module build (step D10).
"""
