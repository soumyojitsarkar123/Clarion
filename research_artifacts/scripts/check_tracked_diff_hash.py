import subprocess,hashlib
b=subprocess.check_output(['git','diff','--binary']);print(hashlib.sha256(b).hexdigest())
