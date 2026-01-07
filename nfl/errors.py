
class NFLError(Exception):
	pass

class NFLTiebreakerError(NFLError):
    pass

    def __init__(self, msg, teams, rule):
    	super().__init__(msg)
    	self.teams = teams
    	self.rule = rule

    def __str__(self):
    	return '{} (teams: {}, last rule: {})'.format(self.args[0], ','.join(self.teams), self.rule)

class NFLScenarioError(NFLError):
	pass
