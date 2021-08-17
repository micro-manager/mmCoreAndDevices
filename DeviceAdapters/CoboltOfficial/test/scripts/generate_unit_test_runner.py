# Unit Test Runner Generator

import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

runnerName = "~runner.cpp"
cxxScript = "..\\cxxtest\\bin\\cxxtestgen"
testRoot = "..\\testsuites"
outputFile = "..\\" + runnerName

def getTestFiles(dir):
    result = []
    for dirName, subdirs, files in os.walk(dir):
        for subdir in subdirs:
            result.extend(getTestFiles(subdir))
        for file in files:
            result.append(dirName + '\\' + file)

    return result

files = getTestFiles(testRoot)
filesStr = ' '.join(files)
command = cxxScript + ' --error-printer -o ' + outputFile + ' ' + filesStr

print("# Generating functional test runner '" + runnerName + "' with command:")
print(command)

os.system(command)

print("# Functional test runner " + runnerName + " generated.")
