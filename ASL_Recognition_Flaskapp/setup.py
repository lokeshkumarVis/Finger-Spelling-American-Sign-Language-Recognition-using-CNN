import pip
import platform

def install(package):
    pip.main(['install', package])

# Example
if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--device", required=True,
		help="Provide the type of device chosen to run this App. -d windows | -d linux | -d raspberrypi")
	args = vars(ap.parse_args())
	device=args["device"]
	if device=='windows':
		py_version=platform.python_version()
		py_version=py_version.split('.')
		if (int(py_version[0])==3) and (int(py_version[1])>4):
			requriements=open('requirements.txt').read()
			requriements=requriements.split('\n')
			for package in requriements:
				if package!='':
					install(package)
			print('Installation Sucessful!!')
	elif device=='raspberrypi':
		print('[WARN] Opencv installation may fail using this setup.')
		print('Alternatively try instaling from source or try "sudo apt-get install python-opencv"')
		py_version=platform.python_version()
		py_version=py_version.split('.')
		if (int(py_version[0])==3) and (int(py_version[1])>4):
			requriements=open('requirements.txt').read()
			requriements=requriements.split('\n')
			for package in requriements:
				if package!='':
					if package=='tensorflow':
						install('tensorflow-1.5.0-cp35-cp35m-linux_armv7l.whl')
					elif package=='opencv-contrib-python':
						try:
							install(package)
						except:
							print('[ERROR] Opencv Installation Failed')
							break
					else:
						install(package)
	elif device=='linux':
		print('[WARN] Opencv installation may fail using this setup.')
		print('Alternatively try instaling from source or try "sudo apt-get install python-opencv"')
		py_version=platform.python_version()
		py_version=py_version.split('.')
		if (int(py_version[0])==3) and (int(py_version[1])>4):
			requriements=open('requirements.txt').read()
			requriements=requriements.split('\n')
			for package in requriements:
				if package!='':
					if package=='opencv-contrib-python':
						try:
							install(package)
						except:
							print('[ERROR] Opencv Installation Failed')
							break
					else:
						install(package)
	else:
		print('[ERROR] Please install Python version 3.5 or later')
