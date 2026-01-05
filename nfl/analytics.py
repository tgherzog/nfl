
import pandas as pd
import numpy as np
from .utils import vmap, ivmap, is_listlike
from .sequence import reorder, expand, expand_len, expand_width
import math

class NFLTiebreakerError(Exception):
    pass

class NFLScenario(pd.Series):
    '''A series of game outcomes created by NFLScenarioMaker. Values will
       be one of 'win', 'loss' or 'tie', indexed by values in the *master*
       games database
    '''

    def __init__(self, host, data, index, name=None):
        super().__init__(data, index=index, name=name, dtype=str)
        self.host = host

    def weeks(self, team):
        '''Return a Series of outcomes for the specified team, one row per week
        '''

        x = self.host.games.join(self.rename('outcome'))
        return x[(x['ht']==team) | (x['at']==team)]['outcome']

    def st(self, team, outcome='win', op='count'):
        '''Test summarized outcomes for a given team.

           team     team code

           outcome  outcome to count

           op       operation - one of:

                    count: returns count of outcome
                    all:   True if all weeks equal outcome, else False
                    any:   True if any week equals outcome, else False

           The result will in all cases be a scalar
        '''

        z = self.weeks(team) == vmap(outcome)
        if op == 'all':
            return z.all()
        elif op == 'any':
            return z.any()

        return z.sum()

    def to_frame(self):
        '''Converts an NFLScenario of game outcomes to a Series indexed by week,team.
           The returned Series will have the same index structure as weekly columns in an NFLScenarioFrame.
           For example:

           with NFLScenarioMaker(nfl, 'NFC-North', -1) as gen:
                outcomes = gen.frame(['outcome'])
                for scenario in gen:
                    x = len(outcomes)
                    outcomes.loc[x] = scenario.to_frame()
        '''

        s = pd.Series('', index=self.host.incomplete, dtype=str, name='outcome')

        # this object must have the same index as self.host.games
        for (k,row) in self.host.games.iterrows():
            result = self.get(k, '')
            if (row['wk'],row['ht']) in s.index:
                s[(row['wk'],row['ht'])] = result

            if (row['wk'],row['at']) in s.index:
                s[(row['wk'],row['at'])] = self.host.aresults[result]

        return s

    def to_wlt(self):
        '''Converts an NFLScenario to a win-loss-tie summary consistent with what NFL.wlt returns
        '''

        z = pd.DataFrame(0, columns=['win','loss','tie'], index=self.host.teams)
        for (k,row) in self.items():
            gm = self.host.games.loc[k]
            if gm['ht'] in z.index:
                z.loc[gm['ht'], row] += 1

            if gm['at'] in z.index:
                z.loc[gm['at'], self.host.aresults[row]] += 1

        return z

        # z = self.to_frame().reset_index().groupby(['team', 'outcome']).count().unstack().xs('week', axis=1)
        # x = pd.DataFrame(0, columns=['win','loss','tie'], index=z.index)
        # x.loc[:, z.columns] = z
        # return x

class NFLScenarioFrame(pd.DataFrame):
    '''A special DataFrame used for analyzing scenarios. Columns are
       a 2-level MultiIndex consisting of weeks in level 0 and teams in level 1.
       Additional columns are typically also included, so long as their
       level-0 columns names are strings not ints, which is now the class
       identifies week columns
    '''
    
    @property
    def wt_len(self):
        '''Returns the number of week/team columns in the dataframe
        '''

        level = 0 if self.columns.names[0] == 'week' else 1
        for n in range(len(self.columns)):
            if type(self.columns[n][level]) is not int:
                return n

        return len(self.columns)

    @property
    def I(self):
        '''Return the dataframe with the week/team levels inverted
        '''

        # week columns are assumed to be at the beginning of the dataframe
        n = self.wt_len
        return self.iloc[:, :n].reorder_levels([1,0], axis=1).sort_index(axis=1).join(self.iloc[:, n:].reorder_levels([1, 0], axis=1).sort_index(level=0, axis=1,sort_remaining=False))

    
    @property
    def X(self):
        '''Return an extended frame with summary info
        '''

        level = 0 if self.columns.names[0] == 'team' else 1
        n = self.wt_len
        teams = self.columns[:n].get_level_values(level).unique()
        elems = ['win','loss','tie']
        x = pd.DataFrame(columns=pd.MultiIndex.from_product([elems, teams], names=['week','team']), index=self.index)
        for elem in elems:
            for t in teams:
                x[(elem, t)] = self.st(t, elem)

        if (self.iloc[:, :n]=='tie').any(axis=None) == False:
            x.drop(columns='tie', level=0, inplace=True)

        if level == 0:
            return self.iloc[:, :n].join( self.iloc[:, n:].join(x.reorder_levels([1,0], axis=1)).sort_index(level=0, axis=1, sort_remaining=False) )

        return self.join(x)


    def weeks(self, team):
        '''Return weeks for the specified team
        '''

        # team/week columns could be inverted
        level = 0 if self.columns.names[0] == 'team' else 1
        n = self.wt_len
        return self.iloc[:, :n].xs(team, level=level, axis=1)

    def test(self, team, *values):
        '''Apply tests by week to a given team

           team        Team code
           
           values      Test values for each week for that team ('win'|1,'loss'|-,'tie'|-1). Use
                       None to ignore a given week

           The return is a DataFrame with the same index as the object and columns for each week
        '''

        z = self.weeks(team)

        c = list(z.columns)

        # a terse way to pad with None and trim to length
        v = (list(values) + ([None] * (len(c)-len(values))))[:len(c)]

        # now eliminate values and columns that are None
        (c,v) = zip(*filter(lambda x: x[1] is not None, zip(c,v)))

        c = list(c)
        v = list(map(lambda x: vmap(x), v))

        return z[c].eq(v).all(axis=1)

    def st(self, team, outcome='win', op='count'):
        '''Test summarized outcomes for a given team

           team     team code

           outcome  outcome to count

           op       operation - one of:

                    count: returns count of outcome
                    all:   True if all weeks equal outcome, else False
                    any:   True if any week equals outcome, else False

           The return is a Series of either counts or Booleans with the same index as the object
        '''

        z = self.weeks(team) == vmap(outcome)
        if op == 'all':
            return z.all(axis=1)
        elif op == 'any':
            return z.any(axis=1)

        return z.sum(axis=1)


class NFLScenarioMaker():
    '''Facilitates generating and testing different win/lose/tie scenarios,
       typically to analyze the outcome on the playoff picture.

       If impelemented in a "with" context the object saves and restores
       the game state.

       Typical usage:

       weeks = [17, 18]
       teams = ['LAC', 'DEN', 'CIN', 'IND', 'MIA']
       with NFLScenarioMaker(nfl, weeks, teams, ties=False) as s:
          df = s.frame(['playoffs'])
          for option in s:
             z = len(df)

       NFLScenarioMaker is an iterable that you can use in wrappers like this:

       from tqdm import tqdm     # progress bar wrapper
       with NFLScenarioMaker(nfl, 'NFC-North', -1) as gen, tqdm(gen) as tgen:
          for scenario in tgen:
            ...
    '''

    def __init__(self, nfl, teams, weeks, ties=True):
        self.nfl = nfl
        self.weeks = list(nfl._weeks(weeks))
        self.teams = list(nfl._list(teams))
        self.ties = ties
        self.stash = None
        self.games = None
        self.completed = None
        self.incomplete = None

        # fixed array, used by related classes, declared here for efficiency
        # defines 'opposing' outcomes
        self.aresults = {'win': 'loss', 'loss': 'win', 'tie': 'tie'}


        # all resulting DataFrames and Series should be sorted by team and week
        self.teams.sort()
        self.weeks.sort()

    def __enter__(self):
        self.stash = self.nfl.stash()

        # get scope of games in this run
        gm = self.nfl.games_
        gm = gm[(gm.index.get_level_values(0)==self.nfl.season) & gm['wk'].isin(self.weeks) & \
                    (gm['ht'].isin(self.teams) | gm['at'].isin(self.teams)) & (gm['p']==False)]

        # The games index defines the structure for each scenario
        self.games = gm[['wk','at','ht']]

        # complete and incomplete are transformations of games.index, structured by [week,team]
        sch = self.nfl.schedule(self.teams, self.weeks, by='team')
        self.completed = sch.dropna().index
        self.incomplete = sch.index.drop(self.completed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.nfl.restore(self.stash)

    def __len__(self):
        types = 2
        if self.ties:
            types += 1

        return types ** len(self.games)

    def __iter__(self):
        '''Iterate over possible scenarios
        '''

        # subroutine based on this:
        # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363
        # 12/24: new implementation using the "yield from" construct, which returns a generator
        # instead of a potentally very large double array
        def outcomes(arrays, z=[]):

            x = arrays[0]
            if len(arrays) > 1:
                for elem in x:
                    yield from outcomes(arrays[1:], z+[elem])
            else:
                for elem in x:
                    yield z + [elem]

        s = NFLScenario(self, 0, index=self.games.index, name='outcome')
        values = ('win','loss','tie')
        if not self.ties:
            values = values[:-1]

        for row in outcomes([values] * len(self.games)):
            s[:] = row
            yield s
            # yield NFLScenario(self, row, index=self.games.index)

    def frame(self, extra=[]):
        '''Return a DataFrame structured to hold the set of scenarios. Typically
           you call this in advance of iterating over the object.

           extra specifiies additional level-0 column names.

           The resulting DataFrame will have a MultiIndex column that is the product
           of weeks and teams, plus an extra set of team columns for each 1st-level
           name in extra. For example:

           >>> with NFLScenarioMaker(nfl, ['NE', 'BUF'], [17,18]) as gen:
           ...   df = gen.frame(['divchamp'])
           ...   df.loc[0] = np.nan # each row defines a unique scenario and outcome
           ...   print(df)
           ... 
           week  17      18     divchamp    
           team  NE BUF  NE BUF       NE BUF
           0    NaN NaN NaN NaN      NaN NaN

           Typical example:

           teams = {'DEN', 'MIA', 'LAC'}
           weeks = [17, 18]

           with nfl.scenarios(weeks, teams) as s:
               df = s.frame(['playoff'])

               # NB: computationally intense: up to 729 possibilities, could take 10m or more
               for option in s:
                  at = len(df)
                  df.loc[at] = option.to_frame()
                  nfl.loc[at, 'playoff'] = ''
                  p = nfl.AFC.playoffs()
                  z = [('playoff', i) for i in set(teams) & set(p.index)]
                  nfl.loc[at, z] = 'x'
        '''

        df = NFLScenarioFrame(columns=pd.MultiIndex.from_product([self.weeks+extra, self.teams], names=['week', 'team']))
        return df.drop(self.completed, axis=1)


class NFLTeamMatrix(pd.DataFrame):
    '''This class specifies a team matrix with potential outcomes.
       You can use this in calls to NFL.set to analyze potential
       end-of-season scenarios.

       You initialize the object with list of teams define both
       the columns and index of a dataframe. The generic code '_'
       is added as the last element in both directions to signify
       an otherwise unspecified team. Note that team order is important,
       as some class methods assume that teams are in rank order
       with "front-runners" first
    '''

    def __init__(self, teams):
        '''Initialize the matrix with a list of teams. Note that division and conference
           codes are *not* supported, so you must pass explicit list-likes of team codes
           in the order you want them defined
        '''

        t = list(teams) + ['_']
        super().__init__(0, columns=t, index=t, dtype='int')
        self.columns.name = 'opp'
        self.index.name = 'name'

    def set_score(self, team, score):
        '''Set team to the specified score

           team is a team code, or '_' for the wildcard

           Pass an integer for score to set the score for all teams without specifying an opponent
           Pass 'win', 'loss' or 'tie' to also set the opponent to the corresponding outcome
        '''

        if team not in self.index:
            raise ValueError('{} is not in the matrix'.format(team))

        if score in ['win','loss','tie']:
            score = ivmap(score)
            self.loc[team] = max(score,0)
            self[team] = 0 if score < 0 else 1-score

    def score(self, team, opp):
        '''Return a tuple consisting of the prescribed outcome (scores) for the two
           specified teams. If either of the teams is not in the matrix then the generic
           '_' outcome is returned. If neither of the teams is specified, or team
           and opp are the same, then an empty tuple is returned
        '''
        if team not in self.index:
            team = '_'

        if opp not in self.index:
            opp = '_'

        if team == opp:
            return ()

        return (self.loc[team,opp],self.loc[opp,team])


    def clean(self, value=0):
        '''Clean the matrix, i.e. set the diagonal to 0 - purely cosmetic
        '''

        for t in self.index:
            self.loc[t, t] = value

    def best(self, team):
        '''Set matrix to the "best possible world" scenario for the specified team.
           This assumes that teams in the matrix are in rank order; teams at the top
           will be handicapped to the advantage of the one specified.

           This scenario provides that:

           1) the specified team wins against all opponents
           3) teams ranked higher than the one specified score 0 points against each other,
              resulting in tie games
           3) Teams ranked lower than the one specified win against higher ranked ones
           4) All teams but the one specified lose to unspecified teams
        '''

        self[:] = 0
        self.set_score('_', 'win')
        self.set_score(team, 'win')
        a = self.index.get_loc(team)
        b = len(self.index)-1

        # teams ranked below beat teams ranked above
        self.iloc[a+1:b, :a] = 1
        self.iloc[:a, a+1:b] = 0
        self.clean()

    def worst(self, team):
        '''Set matrix to the "worst possible world" scenario for the specified team.
           This assumes that teams in the matrix are in rank order; teams at the bottom
           will be handicapped to the disadvantage of the one specified.

           This scenario provides that:

           1) the specified team loses against all opponents
           2) teams ranked lower than the one specified beat all opponents beneath them
        '''

        self[:] = 1
        a = self.index.get_loc(team) + 1
        for n in range(a+1, len(self.index)-1):
            self.iloc[n+1:, n] = 0
            self.iloc[:a, n] = 0

        self.set_score('_', 'loss')
        self.set_score(team, 'loss')
        self.clean()

    @property
    def teams(self):
        '''Return matrix teams as a set
        '''

        return set(self.columns[:-1])

class NFLGameMatrix(pd.DataFrame):
    '''This class provides a team matrix of game records. Each team's row
       contains the number of wins against the opponent in each column. Conversely,
       Each column contains the number of losses against the opponent in each
       row. Ties are tallied as half a win and half a loss. The sum of the two
       numbers is the number of games played by each team combination.

       This class is typically generated by NFL.matrix
    '''

    def __init__(self, teams=None, data=None):
        '''Initialize the matrix with a list of teams. Note that division and conference
           codes are *not* supported, so you must pass explicit list-likes of team codes
           in the order you want them defined

           data can be a DataFrame used to provide initial values. columns will be reindexed
           using teams if provided
        '''

        if data is None and teams is None:
            raise ValueError('NFLGameMatrix must be initialized with a team list and/or data')
        elif data is None and teams is not None:
            t = list(teams)
            super().__init__(0.0, columns=t, index=t, dtype='float32')
        elif teams is None and data is not None:
            super().__init__(data=data, dtype='float32', copy=True)
        else:
            t = list(teams)
            super().__init__(data=data, columns=t, index=t, dtype='float32', copy=True)

        self.columns.name = 'opp'
        self.index.name = 'name'

        # set diagonals to NaN
        z = range(len(self))
        self.iloc[(z, z)] = np.nan

    def games(self):
        '''Transform the matrix into a count of games between each combination of teams
           The numbers will be the same across the centerline of the matrix 
        '''

        m = self.copy()
        for i in range(len(m.columns)-1):
            for n in range(i+1, len(m.columns)):
                s = m.iloc[i, n] + m.iloc[n, i]
                m.iloc[i, n] = s
                m.iloc[n, i] = s

        return m

    def same(self):
        '''Return True if all teams in the matrix have played each other the same number
           of games
        '''

        if len(self) > 1:
            m = self.games()
            return m.isin([np.nan, m.iloc[0,1]]).all(axis=None)

    def pct(self, team, opp=None):
        '''Return the wlt percentage of a team against a specified opponent

           opp: the opposing team. If none, then return the pct for the
           win percentage against all other teams (or None)
        '''

        if opp is None:
            w = self.loc[team].sum()
            g = w + self[team].sum()
            if g > 0:
                return w / g

        elif team != opp and team in self.columns and opp in self.columns:
            w = self.loc[team, opp]
            g = w + self.loc[opp, team]
            if g > 0:
                return w / g

    def sweep(self):
        '''Returns a tuple of a "head-to-head sweep" analysis, which is the
           team that has either beaten all others or lost to all others. The
           return is a tuple (team,0|1) or (None,None)
        '''

        # NB: it's not logically possible that there can be multiple
        # winning sweep teams or losing sweep teams, assuming that every
        # team plays the others at least once

        for t in self.columns:
            wins   = self.loc[t].dropna()
            losses = self[t].dropna()
            if (wins > 0).all() and losses.sum() == 0:
                # all wins, no losses or ties
                return (t, 1)
            elif (wins == 0).all() and (losses > 0).all():
                return (t, 0)

        return (None,None)

    def submatrix(self, teams):
        ''' Return a submatrix of the current matrix for the specified teams
        '''

        if type(teams) is set:
            teams = list(teams)        

        return self.loc[teams, teams]

class NFLTiebreakerController(object):

    def __init__(self, nfl, teams):
        self.nfl = nfl
        self.cg = None
        self.tb_rules = {}
        self.tb_cache = {}
        self.tb = None
        self.gm = None
        self.teams = nfl._list(teams)

        for i in ('div','conf'):
            self.tb_rules[i] = self.get_rules(i)

    def get_rules(self, n):
        '''Return the rule list for category n (div|conv)
        '''

        rules = ['overall','head-to-head']

        if n == 'div':
            rules += ['division', 'common-games', 'conference']
        else:
            rules += ['conference', 'common-games']

        rules += ['victory-strength', 'schedule-strength', 'conference-rank', 'overall-rank']

        if rules[2] == 'division':
            rules.append('common-netpoints')
        else:
            rules.append('conference-netpoints')

        rules.append('overall-netpoints')

        if self.nfl.netTouchdowns:
            rules.append('net-touchdowns')

        return rules

    def tiebreaker(self, teams, winner=None, rule=''):
        '''Get or set a tiebreaker outcome from cache
        '''

        t = list(teams)
        t.sort()
        k = ':'.join(t)
        if winner:
            self.tb_cache[k] = (winner,rule)
        else:
            return self.tb_cache.get(k, (None,None))

    def tiebreakers(self):
        '''Return a tiebreaker DataFrame for the object's teams, ideally from cache
        '''

        if self.tb is None:
            self.tb = self.nfl.tiebreakers(self.teams, strict=False, controller=self).xs('pct', axis=1, level=1).T

        return self.tb

    def gamematrix(self):
        '''Return an NFLGameMatridx for the specified teams, ideally from cache
        '''

        if self.gm is None:
            self.gm = self.nfl.matrix(self.teams)

        return self.gm


    def rules(self, teams):
        '''Return tiebreaker rules for the given teams, in hierarchical order
           teams may also be a keyword, 'div' or 'conf'
        '''
        if type(teams) is str:
            k = 'div' if teams == 'div' else 'conf'
        else:
            k = 'div' if len(set( map(lambda x: self.nfl.teams_[x]['div'], teams) )) == 1 else 'conf'

        return self.tb_rules[k]

    def common_games(self, teams):
        ''' Return the wlt DataFrame for the given teams against their common
            opponents
        '''

        t = teams.copy()
        t.sort()
        k = ':'.join(t)

        if self.cg is not None and k in self.cg.get_level_values(0):
            return self.cg.xs(k)
        
        s = self.nfl.wlt(teams, within=self.nfl.opponents(teams), result='long')
        s = s.assign(key=k).set_index('key', append=True).swaplevel()
        if self.cg is None:
            self.cg = s
        else:
            self.cg = pd.concat([self.cg, s])

        return self.cg.xs(k)


class NFLSequenceMaker():

    def __init__(self, source, n=1):
        self.source = source
        self.n = n

    def __len__(self):

        return expand_len(self.source, self.n) * math.factorial(expand_width(self.surce, self.n))

    def __iter__(self):

        for elems in self.expand():
            yield from self.reorder(elems)

    def reorder(self, elems):
        yield from reorder(elems)


    def expand(self):
        yield from expand(self.source, self.n)
