print("Verifying imports and versions")

try:
    import numpy as np
    print("== numpy OK ==")
except ImportError:
    print("== numpy NOT installed ==")

try:
    import hnswlib
    print("== hnswlib OK ==")
except ImportError:
    print("== hnswlib NOT installed ==")

try:
    from docarray.index import HnswDocumentIndex
    print("== docarray y HnswDocumentIndex OK ==")
except ImportError as e:
    print("== docarray or HnswDocumentIndex NOT available ==")
    print(" Error:", e )

try:
    import vectordb
    print("== vectordb OK ==")
except ImportError:
    print("== vectordb NOT installed ==")

try:
    import jina
    print("== jina OK ==")
except ImportError:
    print("== jina NOT installed ==")

try:
    import matplotlib
    print("== matplotlib OK ==")
except ImportError:
    print("== matplotlib NOT installed ==")

print(" [INFO] Verification completed")
