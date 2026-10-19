
import pandas as pd
import numpy as np
from .utils import vmap, ivmap, is_listlike, param_exc
from .errors import *
from .sequence import reorder, expand, expand_len, expand_width
import math


class NFLScenario(pd.Series):
    '''A series of game outcomes in an NFLScenarioFrame
    '''

    _metadata = ['nfl', 'schedule']

    @property
    def _constructor(self):
        return NFLScenario

    @property
    def _constructor_expanddim(self):
        def f(*args, **kw):
            return NFLScenarioFrame.make(self.nfl, self.schedule, *args, **kw)

        return f

    @property
    def _external(self):
        '''Return self with an index that matches nfl.games_. This ensures
           that index column names match and that any custom rows have been removed.
           Otherwise join typically fails and you may raise KeyErrors
        '''
        return self.reindex(self.index.set_names(['seas','id']))[self.index.get_level_values(0)=='reg']

    @staticmethod
    def from_games(nfl, teams=None, weeks=None, index=None):
        '''Create an NFLScenario from the current game database. This is 
           intended for troubleshooting analysis and isn't really relevant elsewhere.
           It can only be used on the regular season

           teams and weeks specify the domain of games for the scenario.
           Alternatively, index (e.g. an existing NFLScenario or NFLScenarioFrame)
           can be specified
        '''

        if isinstance(index, NFLScenario):
            gm = nfl.games_.loc[index.index]
            sch = index.schedule

        elif isinstance(index, NFLScenarioFrame):
            gm = nfl.games_.loc[index.columns]
            sch = index.schedule

        elif index is not None:
            gm = nfl.games_.loc[index]

            # infer schedule from the game database
            ht = gm.rename(columns={'wk': 'week', 'ht': 'team'}).set_index(['week', 'team'])['p']
            at = gm.rename(columns={'wk': 'week', 'at': 'team'}).set_index(['week', 'team'])['p']
            sch = pd.concat([ht, at]).sort_index().index

        else:
            teams = nfl._list(teams)
            weeks = nfl._weeks(weeks)
            gm = nfl.games_
            gm = gm[gm.index.get_level_values(0)=='reg']
            if teams is not None:
                gm = gm[gm['at'].isin(teams) | gm['ht'].isin(teams)]

            if weeks is not None:
                gm = gm[gm['wk'].isin(weeks)]

            sch = nfl.schedule(teams, weeks, by='team', season='reg').index

        s = NFLScenario(gm.apply(lambda x: 1 if x['hs'] > x['as'] else (0 if x['hs'] < x['as'] else -1), axis=1), dtype='int8')
        s.nfl = nfl
        s.schedule = sch
        return s


    def weeks(self, team):
        '''Return a Series of outcomes for the specified team, one row per week
        '''

        return self.by('week').loc[:, team]

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

    def by(self, how):
        '''Returns the populated schedule for this scenario

           If how is 'week' the schedule is sorted by week then team. If
           'team' then the opposite
        '''

        if how not in ['week','team']:
            raise param_exc('how', how)

        z = pd.Series('', index=self.schedule)
        s = self._external
        g = self.nfl.games_.loc[s.index].join(s.rename('o')).rename(columns={'wk': 'week'})

        ht = g.rename(columns={'ht': 'team'}).set_index(['week','team'])['o'].map({1: 'win', 0: 'loss', -1: 'tie'})
        at = g.rename(columns={'at': 'team'}).set_index(['week','team'])['o'].map({1: 'loss', 0: 'win', -1: 'tie'})

        z.update(ht)
        z.update(at)

        if how == 'team':
            z = z.swaplevel(1,0).sort_index()

        return z.rename('outcome')

    @property
    def wlt(self):
        '''Converts the win-loss-tie summary for this scenario
        '''

        return self.by('week').reset_index().groupby(['team','outcome'])['week'].count().unstack().fillna(0).astype('int16')

class NFLScenarioFrame(pd.DataFrame):
    '''A special DataFrame used for analyzing scenarios. Columns are
       a 2-level MultiIndex that aigns with the index in nfl.games_
       Additional columns may be added
    '''

    _metadata = ['nfl', 'schedule']

    @staticmethod
    def make(nfl, schedule, *args, **kw):
        df = NFLScenarioFrame(*args, **kw)
        df.columns = df.columns.set_names(['cat','code'])
        df.nfl = nfl

        # schedule is a 2-level index of weeks and teams used to generate this
        # DataFrame, and is used in transforms
        df.schedule = schedule
        return df

    @property
    def _constructor(self):
        return NFLScenarioFrame

    @property
    def _constructor_sliced(self):
        def f(*args, **kw):
            s = NFLScenario(*args, **kw)

            # somehow, NFLScenarioFrame objects are getting allocated or copied
            # without their attributes, so these checks are necessary. It seems
            # to be under the __repr__ chain, where they are non-critical
            # If missing, I'm marking these with a string to help troubleshoot
            if hasattr(self, 'schedule'):
                s.schedule = self.schedule
            else:
                s.schedule = 'error'

            if hasattr(self, 'nfl'):
                s.nfl = self.nfl
            else:
                s.nfl = 'error'

            return s

        return f


    def by(self, how):
        '''Return the object with games replaced with NFLScenario.by() results
           This is expensive; best to use on small slices
        '''

        x = len(self.columns[self.columns.get_level_values(0)=='reg'])
        if x < len(self.columns):
            return pd.DataFrame( self.apply(lambda s: pd.concat([s.by(how), s.iloc[x:]]), axis=1) )

        # downcast this back to DataFrame since rows no longer function as NFLScenarios
        return pd.DataFrame( self.apply(lambda s: s.by(how), axis=1) )


    def S(self, at):
        ''' Return the NFLScenario at the given row using self.loc[]. This option
            preserves the original Series structure, which will otherwise be up-cast
            if other columns are added to the DataFrame
        '''

        return self.loc[at, self.columns.get_level_values(0)=='reg']

    def T(self, team, value=None):
        '''Return games for the specified team and optionally apply outcome tests

           If value is None then the games for the specified team are returned
           in week order. Values will be mapped to outcomes from the specified
           team's perspective: 1 indicates a win, 0 a loss, -1 a tie

           If value is a single value then it is evaluated against the specified
           games and a single Series is returned indicating which rows match
           for all games. 'win', 'loss', 'tie' can be
           specified in lieu of 1, 0, -1

           If value is a list-like then you can specify game-specific values,
           e.g. [1, 0, 1] or ['win', 'loss', 'win']. None values in the list
           are treated like wildcards

           This method is designed to make sub-setting easier

           Examples:

           # return outcomes for the Packers
           df.T('GB')

           # just scenarios where the Bears run the table
           df[ df.T('CHI' 'win') ]

           # just scenarios where the Lions win the the 1st game and lose the 3rd
           df[ df.T('DET', ['win', None, 'loss']) ]

           # just scenarios where the Vikings win exactly 2 games
           df[ (df.T('MIN')==1).sum(axis=1)==2 ]
        '''

        x = len(self.columns[self.columns.get_level_values(0)=='reg'])
        gm = self.nfl.games_.loc[self.columns[:x]]
        gm = gm[(gm['at']==team) | (gm['ht']==team)]

        vmap = {'win': 1, 'loss': 0, 'tie': -1}

        df = self.loc[:, gm.index] # avoid SettingWithCopyWarning
        for elem in gm[gm['at']==team].index:
            df[elem] = df[elem].replace({1:0, 0:1})

        if value is None:
            return df
        elif type(value) is not str and hasattr(value, '__iter__'):
            # assumed to be an array of matching values - None to omit
            c = []
            v = []
            n = 0
            for x in value:
                if n >= len(df.columns):
                    break

                if x is not None:
                    c.append(df.columns[n])
                    v.append(vmap.get(x,x))

                n += 1

            return (df[c] == v).all(axis=1)

        # else assumed to be a scalar
        return (df == vmap.get(value, value)).all(axis=1)


class NFLSeasonWrapper():
    '''This class creates an context that changes the default season on
       an NFL object. This makes it easier and safer to perform calculations
       where the season must be hard-coded, for instance, calculating playoff
       scenarios and tiebreakers.  It provides an alternative to always setting
       the season parameter explicitly on function calls. Implementation looks like this:

       nfl.season = 'post' # schedules, records, etc default to post-season
       with NFLSeasonWrapper(nfl, 'reg'):
          # nfl.season is now 'reg'

       # nfl.season is back to post
    '''

    def __init__(self, nfl, season='reg'):
        self.nfl = nfl
        self.season = season

    def __enter__(self):
        self._season = self.nfl.season
        self.nfl.season = self.season

    def __exit__(self, exc_type, exc_value, traceback):
        self.nfl.season = self._season

class NFLStateWrapper():
    '''This class creates an context that stashes the current state of an NFL object,
       making it easier to perform an analysis and restore the state afterwards.
       This is an alternative to explicitly calling NFL.stash() and NFL.restore()
       an NFL object. This makes it easier and safer to perform calculations
       where the season must be hard-coded, for instance, calculating playoff
       scenarios and tiebreakers.  It provides an alternative to always setting
       the season parameter explicitly on function calls. Implementation looks like this:

       nfl.season = 'post' # schedules, records, etc default to post-season
       with NFLStateWrapper(nfl):
          # perform operations on games and statistics

       # nfl games and statistics are restored
    '''

    def __init__(self, nfl):
        self.nfl = nfl

    def __enter__(self):
        self.state = self.nfl.stash()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.nfl.restore(self.state)

class NFLEmptyContextWrapper():
    ''' An empty wrapper to use in code requiring a context object when none
        is specified by the user. Typically this is used as a fallback for
        code that provides optional support for tqdm or other progress reporting toos
    '''

    def __init__(self, obj, **kwargs):
        self.obj = obj
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __iter__(self):
        return self.obj.__iter__()


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
        self.index.name = 'team'

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
           3) teams ranked above the one specified score 0 points against each other,
              resulting in tie games
           3) Teams ranked lower than the one specified win against higher ranked ones
           4) All teams but the one specified lose to unspecified opponents
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
           2) teams ranked above the one specified score 1 point against each other,
              resulting in ties
           3) teams ranked below the one specified beat all teams but lose against those
              ranked between themselves and the one specified
           4) All teams but the one specified beat unspecified opponents
        '''

        self[:] = 1
        a = self.index.get_loc(team) + 1
        for n in range(a, len(self.index)-1):
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

    @property
    def _constructor(self):
        return NFLGameMatrix

    @staticmethod
    def make(teams=None, data=None):
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
            df = NFLGameMatrix(0.0, columns=t, index=t, dtype='float32')
        elif teams is None and data is not None:
            df = NFLGameMatrix(data=data, dtype='float32', copy=True)
        else:
            t = list(teams)
            df = NFLGameMatrix(data=data, columns=t, index=t, dtype='float32', copy=True)

        df.columns.name = 'opp'
        df.index.name = 'name'

        # set diagonals to NaN
        z = range(len(df))
        df.iloc[(z, z)] = np.nan
        return df

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

class NFLTiebreakerController(object):

    def __init__(self, nfl, teams, limit=None):
        self.nfl = nfl
        self.tb_rules = {}
        self.teams = list(nfl._list(teams))

        self.reset() # initialize cache properties

        for i in ('div','conf'):
            self.tb_rules[i] = self.get_rules(i)

        if limit:
            common = self.common_rules[1:]
            if limit not in common:
                raise NFLScenarioError("limit was {}, must be one of {}".format(limit, ', '.join(common)))

            for k,v in self.tb_rules.items():
                self.tb_rules[k] = v[:v.index(limit)]


    def reset(self, reload=False):
        '''Reset any and all internal caches

           Set reload to True to reload caches immediately (otherwise, lazy load)
        '''

        self.tb_cache = {}
        self.stats = None
        self.gm = None

        if reload:
            self.stats = self.nfl._stats().loc[self.teams]
            self.gm = self.nfl.matrix(self.teams, season='reg')

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

        if self.nfl.config.netTouchdowns:
            rules.append('net-touchdowns')

        return rules

    def stat(self, rule, teams, df=False):
        ''' Return statistics for a specified rule. This should be a Series indexed by team
        '''

        # rule/column mappings for when they differ
        lookup = {'victory-strength': 'vic_stren', 'schedule-strength': 'sch_stren',
                    'conference-rank': ('misc', 'rank-conf'), 'overall-rank': ('misc','rank-overall')}

        def result(obj):
            if df:
                if isinstance(obj, pd.Series):
                    return pd.DataFrame({'win': np.nan, 'loss': np.nan, 'tie': np.nan, 'pct': obj}, index=obj.index, dtype='float32')

                return obj

            # just wants the pct column 
            if isinstance(obj, pd.Series):
                return obj

            return obj['pct']

        stats = self.stats
        if stats is None:
            stats = self.nfl._stats().loc[self.teams]
            self.stats = stats

        if type(teams) is set:
            teams = list(teams) 

        with NFLSeasonWrapper(self.nfl):
            if rule == 'head-to-head':
                if df:
                    # Matrix doesn't give us accurate wlt tallies so we have to calculate
                   return self.nfl.wlt(teams, within=teams)

                gm = self.gamematrix(teams)
                s = pd.Series(np.nan, index=gm.index)
                for i in gm.index:
                    s[i] = gm.pct(i)

                return result(s)

            elif rule == 'common-games':
                return result(self.nfl.wlt(teams, within=self.nfl.opponents(teams, allGames=True)))

            elif rule == 'common-netpoints':
                s = self.nfl.wlt(teams, within=self.nfl.opponents(teams, allGames=True), points=True)
                return result(s['scored'] - s['allowed'])

            elif rule == 'overall-netpoints':
                return result(stats.loc[teams, ('misc','pts-scored')] - stats.loc[teams, ('misc','pts-allowed')])

            elif rule == 'conference-netpoints':
                return result(stats.loc[teams, ('misc','conf-pts-scored')] - stats.loc[teams, ('misc','conf-pts-allowed')])

            elif rule == 'net-touchdowns':
                # TODO: implement team-level caching - this hits the API so it's super slow
                return result(pd.Series(self.nfl.net_stats(teams)))

            else:
                col = lookup.get(rule, rule)
                return result(stats.loc[teams, col])

        

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

    def gamematrix(self, teams):
        '''Return an NFLGameMatridx for the specified teams, ideally from cache
        '''

        if self.gm is None:
            self.gm = self.nfl.matrix(self.teams, season='reg')

        if type(teams) is set:
            teams = list(teams) 

        return self.gm.loc[teams, teams]

    def rules(self, teams, all=False):
        '''Return tiebreaker rules for the given teams, in hierarchical order
           teams may also be a keyword, 'div' or 'conf'
        '''
        if type(teams) is str:
            if teams in self.tb_rules:
                return self.tb_rules[teams]

            raise param_exc('teams', teams)

        k = 'div' if len(set( map(lambda x: self.nfl.teams_[x]['div'], teams) )) == 1 else 'conf'

        if all:
            # return all rules but with order and priority dicated by teams
            (a,b) = (self.rules('div'), self.rules('conf'))
            if k == 'conf':
                (a,b) = (b,a)

            return a + list(set(b) - set(a))

        # else, return rules that match the teams
        return self.tb_rules[k]

    @property
    def common_rules(self):
        '''Return the list of rules common to both contexts. This represents where limits
           can be set
        '''

        a = self.rules('div')
        b = self.rules('conf')

        c = set(a) & set(b)
        return list( filter(lambda x: x in c, a) )

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
