import os
import sys

# make gitignore file if not exists, and add output folders
# if not os.path.exists('../.gitignore'):
#     with open('../.gitignore', 'w') as fp:
#         pass
# with open('../.gitignore') as f:
#     gitignore = f.read()
# with open('../.gitignore', 'a') as f:
#     if 'ed/' not in gitignore:
#         f.write('ed/\n')
#     if 'IS2data' not in gitignore:
#         f.write('IS2data/\n')
#     if 'plots/' not in gitignore:
#         f.write('plots/\n')
#     if 'imagery/' not in gitignore:
#         f.write('imagery/\n')
#     if '*ipynb_checkpoints' not in gitignore:
#         f.write('*ipynb_checkpoints\n')
#     if '*__pycache__' not in gitignore:
#         f.write('*__pycache__\n')

# check for earthdata credentials
if not os.path.exists('ed'):
    os.makedirs('ed')
    with open("ed/__init__.py", "w") as f:
        print('import ed.edcreds', file=f)
    with open("ed/edcreds.py", "w") as f:
        print("def getedcreds():\n", file=f)
        print("\t\n", file=f)
        print("\t# add your nasa earthdata login credentials below\n", file=f)
        print("\tuser_name = '<your_earthdata_username>'\n", file=f)
        print("\tpassword = '<your_earthdata_password>'\n", file=f)
        print("\temail = '<your_email>'\n", file=f)
        print("\t\n", file=f)
        print("\treturn user_name, password, email\n", file=f)
    print("Earthdata user credentials are not saved yet!")
    print("  --> Open ed/edcreds.py and enter your credentials where prompted.")
    print("  --> Then save the file and re-start your kernel if in Jupyterlab.")
    sys.exit()

if not os.path.exists('IS2data'):
    os.makedirs('IS2data')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('imagery'):
     os.makedirs('imagery')
        
import utils.oa
import utils.curve_intersect
import utils.nsidc
import utils.S2