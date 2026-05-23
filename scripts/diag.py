import sys
print("EXEC:", sys.executable)
print("PATH[:3]:", sys.path[:3])
print("CWD:", __import__("os").getcwd())
try:
    import pandas
    print("pandas:", pandas.__version__)
except Exception as e:
    print("pandas FAIL:", e)
try:
    import omni
    print("omni:", omni.__file__)
except Exception as e:
    print("omni FAIL:", e)
