from web import form

#update to accept params which dynamically create the form
class GolfForm:
 	
	def __init__(self):

		self.currentForm = form.Form(

			form.Textbox("Course_Par"),
			form.Textbox("Your_Score", 
				form.notnull,
				form.regexp('\d+', 'Must be a digit')),
				#form.Validator('Must be more than 5', lambda x:int(x)>5)),
			form.Checkbox('Pro'), 
			form.Dropdown('Handicap: ', ['0-9', '10-19', '20-29', '30+']),
			form.Dropdown('Number of putts: ', ['< 25', '26-30', '31-35', '36+']),
			form.Dropdown('Number of fairways hit: ', ['< 5', '5-10', '> 10']),
			form.Dropdown('Number of par 3 greens hit: ', ['0', '1-2', ' > 2']),

			)

	def getMainForm(self):
		return self.currentForm