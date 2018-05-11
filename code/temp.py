import re
import tokenize
code_words = []
def getCode():
	Errorfile = open('./code/new.txt','rb')
	CodeFile = open('./code/temp.java','rb')
	
	# outfile = 
	i = 0
	errorLinePos = Errorfile.readline()
	errorLine =Errorfile.readline()
	errorPos = Errorfile.readline()
	errorLinePos = re.findall(r'\d+',errorLinePos)
	# print errorLine
	# print line2
	errorLine = list(errorLine)
	global code_words
	code_words = []
	errorLine  = errorLine[:len(errorLine)-1]
	errorLine = ''.join(errorLine)
	errorInd =  list(errorPos).index('^')
	print('val',errorLinePos[0],errorInd)
	def printit(a,b,c,d,e):
		global code_words
		print(c[0])
		if(c[0] < int(errorLinePos[0]))  or (c[0] == int(errorLinePos[0]) and d[1] < (errorInd -11)):
			print('heck',b)
			code_words.append(b)
		else:
			print('not happening')	
		# print('{} {} {} {} {} '.format(a,b,c,d,e))
	def sendline():
		global i
		if i is 1:
			return ''
		i= i+1
		return line1
	
	try:
		tokenize.tokenize(CodeFile.readline,printit)
	except Exception as e:
		print(e)
		print("EOF")
	print('words',code_words)	
	# code_words = [i for i in code_words if i != '\n']	
	return [code_words]
