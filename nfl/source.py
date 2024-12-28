
class NFLSource():
    lasturl = None

    def boxscore(self, nfl, code, week):
        '''Return box score as a data frame
        '''

        raise NotImplementedError('{} does not implement box scores'.format(self.__class__.__name__))

    def plays(self, nfl, code, week, count):
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

    def extra_fields(self, type):
        '''Return extra field names requested by this source
        '''
