
class NFLSource():
    '''Base class for NFL data source. This class must implement the games() and teams()
       functions at a minimum to be useful. More details in espn.py
    '''

    def __init__(self):
        self.lasturl = None

    def boxscore(self, nfl, game):
        '''Return box score as a data frame
        '''

        raise NotImplementedError('{} does not implement box scores'.format(self.__class__.__name__))

    def plays(self, nfl, game, count):
        '''Return most recent plays from the specified game
        '''

        raise NotImplementedError('{} does not implement play drives'.format(self.__class__.__name__))

    def drives(self, nfl, game):
        '''Return most recent plays from the specified game
        '''

        raise NotImplementedError('{} does not implement play drives'.format(self.__class__.__name__))

    def scoreboard(self, nfl):
        '''Return live scores as an NFLScoreboard object
        '''

        raise NotImplementedError('{} does not implement scoreboard'.format(self.__class__.__name__))

    def roster(self, nfl, code):
        '''Return team roster as a data frame
        '''

        raise NotImplementedError('{} does not implement team rosters'.format(self.__class__.__name__))

    def net_touchdowns(self, nfl, teams):
        '''Return a dict of net touchdowns for specified teams
        '''

        raise NotImplementedError('{} does not implement net touchdowns'.format(self.__class__.__name__))

