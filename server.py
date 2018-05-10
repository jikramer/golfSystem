import web
from web import form
import scorer
from GolfForm import GolfForm 

render = web.template.render('templates/')

urls = ('/', 'index')

#start the app
app = web.application(urls, globals())

#create the form
mainForm = GolfForm().getMainForm()
 
	
class index: 
    def GET(self): 
	 	scorer.loadRound()							#builds the model on call to '/'
		return render.golfSystemForm(mainForm)		#displays the default form

    def POST(self): 
        if not mainForm.validates(): 
            return render.golfSystemForm(mainForm)	#if invalid, reload initial form
        else:
 			#result = scorer.testLinearRegression()			#call the scorer
			result = scorer.testBayes()			#call the scorer
			
  			rc = "Test success! Course_Par: %s, Your_Score: %s" % (mainForm.d.Course_Par, mainForm['Your_Score'].value)		#render basic results
			rc = rc  +  "\n" + "'Accuracy score' : " + result 
			return rc

if __name__=="__main__":
    web.internalerror = web.debugerror
    app.run()	