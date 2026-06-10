
class NFLError(Exception):
	pass

class NFLTiebreakerError(NFLError):
    pass

    def __init__(self, msg, teams, rules):
    	super().__init__(msg)
    	self.teams = teams
    	self.rules = rules

    def __str__(self):
    	return '{} (teams: {}, rules: {})'.format(self.args[0], ','.join(self.teams), self.rules)

class NFLScenarioError(NFLError):
	pass
