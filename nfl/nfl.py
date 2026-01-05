
import sys
import logging
import copy
from datetime import datetime
import numpy as np
import pandas as pd

from .utils import current_season, vmap, ivmap
from .analytics import NFLTeamMatrix, NFLGameMatrix, NFLScenario, NFLScenarioMaker
from .__version__ import __version__

class NFLTeam():
    '''an NFL team, typically obtained by calling the NFL object with the team code

       Local variables you can use (but not change)

       code: team code: identifies the team in API calls

       name: team name

       div:  division code

       conf: conference code
    '''


    def __init__(self, code, host):

        elem = host.teams_[code]
        self.code = code
        self.name = elem['name']
        self.div  = elem['div']
        self.conf = elem['conf']
        self.host = host

    @property
    def division(self):
        '''Set of teams in this team's division
        '''
        return self.host.divs_[self.div]

    @property
    def conference(self):
        '''Set of teams in this team's conference
        '''
        return self.host.confs_[self.conf]

    @property
    def standings(self):
        '''Team standings
        '''

        return self.host.team_stats(self.code)

    @property
    def schedule(self):
        '''Team schedule
        '''

        return self.host.schedule(self.code, by='team', season='reg')

    @property
    def roster(self):
        '''Team roster
        '''
        return self.host.roster(self.code)

    @property
    def active_game(self):
        '''Returns the most recently active game, if any
        '''

        z = self.host.games_
        z = z[((z['at']==self.code) | (z['ht']==self.code)) & (z['ts'] < datetime.now())]
        if len(z) > 0:
            return z.droplevel(0).iloc[-1]


    def __getitem__(self, key):
        ''' subscript into the player roster
        '''
        return self.roster[key]

    @property
    def boxscore(self):
        '''Boxscore stats for the active game
        '''

        return self.host.boxscore(self.code)

    @property
    def streak(self):

        return self.host.streak(self.code)


    def plays(self, drive=None, count=None, week=None, season=None):
        '''Returns most recent plays from the specified game.

           drive:   drive number (the index from the DataFrame returned by NFLTeam.drives())
           count:   number of most recent plays to return. If None, defaults to 10 unless drive is specified
           week:    game week; otherwise, the current week

           Note that the returned object is a DataFrame wrapper with an extra
           function that lets you more easily read the play description, which
           in the ordinary DataFrame view is typically truncated.

           Examples:
           nfl('MIN').plays() # return most recent plays
           nfl('MIN').plays().desc() # print the description field from the most recent play
           nfl('MIN').plays().desc(-5) # description from 5 plays back
        '''

        if week:
            game = self.host.game(self.code, week, season)
        else:
            game = self.active_game

        if drive is None and count is None:
            count = 10

        if game is not None:
            return self.host.engine.plays(self.host, game, drive, count)

    def drives(self, week=None, season=None):
        '''Returns drive summary from the specified game as a DataFrame

           The dataframe includes two columns that display changes in game score.
           For readability the column names are the same as the team IDs. As a convenience,
           the engine adds 'ateam' and 'hteam' properties to the dataframe to
           make it easier to identify these programatically
        '''

        if week:
            game = self.host.game(self.code, week, season)
        else:
            game = self.active_game

        if game is not None:
            return self.host.engine.drives(self.host, game)
            

    def __repr__(self):
        wk = self.host.week or 1
        s  = self.schedule.loc[1:wk+3].__repr__()
        return '{}: {} ({})\n'.format(self.code, self.name, self.div) + self.standings.__repr__() + '\n' + s

    def _repr_html_(self):
        wk = self.host.week or 1
        s  = self.schedule.loc[1:wk+3]._repr_html_()
        return '<h4>{}: {} ({})</h4>\n'.format(self.code, self.name, self.div) + self.standings._repr_html_() + '\n' + s


class NFLDivision():
    def __init__(self, code, host):

        self.code = code
        self.host = host
        self.teams = host.divs_[code]

    @property
    def standings(self):
        '''Division standings
        '''

        z = self.host.standings
        z = z[z['div']==self.code]
        return z

    def __repr__(self):
        # ensures tiebreakers are used to determine order, then deletes the rank column
        return '{}\n'.format(self.code) + self.standings.__repr__()

    def _repr_html_(self):
        return '<h3>{}</h3>\n'.format(self.code) + self.standings._repr_html_()

class NFLConference():
    def __init__(self, code, host):

        self.code = code
        self.host = host
        self.teams = host.confs_[code]

    @property
    def divisions(self):
        return set(map(lambda x: '-'.join([self.code, x]), ['North', 'East', 'South', 'West']))


    @property
    def standings(self):
        '''Conference standings
        '''

        z = self.host.standings
        return z[z['div'].str.startswith(self.code)]
        return z


    def playoffs(self, count=7):
        '''Current conference seeds, in order

           count     number of seeds to return
        '''

        # get top-listed teams from each division
        champs = self.standings.reset_index().groupby(('div','')).first()['team']

        # restructure and reorder by seed
        champs = pd.Series(map(lambda x: x.split('-')[-1], champs.index), index=champs)
        champs = champs.reindex(self.host.tiebreaks(champs.index).index)

        # append all remaining teams in tiebreaker order
        if count > len(champs):
            champs = pd.concat([champs, pd.Series('', index=self.host.tiebreaks(self.teams-set(champs.index), limit=count-len(champs)).index)])
            i = 1
            for elem in champs[4:7].index:
                champs[elem] = 'Wildcard{}'.format(i)
                i += 1

        return champs[:count]


    def __repr__(self):
        return '{}\n'.format(self.code) + self.standings.__repr__()

    def _repr_html_(self):
        return '<h3>{}</h3>\n'.format(self.code) + self.standings._repr_html_()

class NFLState():
    '''Simple storage class that holds copies of critical information, to be restored later
    '''

    def __init__(self, nfl):
        self.games = nfl.games_.copy()
        self.week  = nfl.week


class NFL():
    '''Impelementation class for NFL data operations

       NFL defines the following instance variables that you can use and change:

       week      Current week number. This is automatically set by NFL.update()
                 and NFL.load(), and is used as a default argument by several functions

       year      Current season. In theory this can be changed work with a previous
                 season, although that's API independent and hasn't really been tested

       season    'pre', 'reg', or 'post' with a default of 'reg'. This is the default
                 value for season-specific functions like schedule, wlt and set/clear. 

       engine    API engine. The ESPN engine is the most robust, but others are
                 available or could be written

       autoUpdate      Determines whether NFLScoreboards automatically update the game database
                       when games complete. Set to False if you don't want this

       netTouchdowns   Determines whether "net touchdowns" are included in tiebreaker determinations
                       Fetching this data from the API is generally time consuming and rarely
                       needed, so leave it set to False in most situations

       display         controls how certain dataframe display teams, either as bare codes or
                       in a more human-readable form. This does not affect dataframe structure
                       in any way. Recognized values are 'user' (human-readable) and 'data' (codes only)
    '''

    def __init__(self, year=None, season=None, engine=None, display='user'):
        self.__version__ = __version__
        self.teams_  = {}
        self.divs_   = {}
        self.confs_  = {}
        self.rosters_ = {}
        self.seasons_ = {}
        self.games_  = None
        self.week = None
        self.stats = None
        self.year = year
        self.season = season or 'reg'
        self.engine = engine
        self.autoUpdate = True
        self.netTouchdowns = False
        self.display = display

        if not self.year:
            self.year = current_season()

        if not engine:
            from .espn import NFLSourceESPN
            self.engine = NFLSourceESPN()


    def __call__(self, i):

        if i in self.confs_:
            return NFLConference(i, self)

        if i in self.divs_:
            return NFLDivision(i, self)

        if i in self.teams_:
            return NFLTeam(i, self)

    def load(self, path):
        ''' Loads data from the specified excel file
        '''

        # in case of reload
        self.path   = path
        self.teams_ = {}
        self.games_ = None

        with pd.ExcelFile(path) as reader:
            self.teams_ = pd.read_excel(reader, sheet_name='teams', index_col=0).to_dict(orient='index')
            self.games_ = pd.read_excel(reader, sheet_name='games', index_col=[0,1])
            meta = pd.read_excel(reader, sheet_name='meta', index_col=0)

        self.year = meta.loc['Year', 'value']
        if not self.week and 'Week' in meta.index:
            self.week = meta.loc['Week', 'value']

        key = meta.loc['Engine', 'value'].split('.')
        engine = getattr(sys.modules['.'.join(key[:-1])], key[-1])
        self.engine = engine()

        # resets and housecleaning
        self._post_load()
        return self

    def save(self, path):
        '''Save data to the specified Excel file
        '''

        teams = pd.DataFrame(columns=['code','name','short','conf','div']).set_index('code')
        meta  = pd.DataFrame(columns=['key', 'value']).set_index('key')

        for key,team in self.teams_.items():
            teams.loc[key, :] = [team['name'], team['short'], team['conf'], team['div']]

        meta.loc['Last Updated', 'value'] = datetime.now()
        meta.loc['Engine', 'value'] = '.'.join([self.engine.__class__.__module__, self.engine.__class__.__name__])
        meta.loc['Year', 'value'] = self.year
        meta.loc['Week', 'value'] = self.week

        with pd.ExcelWriter(path) as writer:
            teams.to_excel(writer, sheet_name='teams')
            self.games_.to_excel(writer, sheet_name='games')
            meta.to_excel(writer, sheet_name='meta')

        self.path = path

    def _stats(self):
        '''Return the master statistics table, building it first if necessary. Teams are ordered by conference,
        division and division rank. Changes to raw game scores (via load, set, etc) invalidate the table
        so the next call to _stats rebuilds it
        '''

        if self.stats is not None:
            return self.stats

        if self.games_ is None:
            raise RuntimeError('game data has not yet been updated or loaded')

        stat_cols = [('name',''),('div',''),('conf','')]
        stat_cols += list(pd.MultiIndex.from_product([['overall','division','conference', 'vic_stren', 'sch_stren'],['win','loss','tie','pct']]))
        stat_cols += list(pd.MultiIndex.from_product([['misc'],['rank-conf','rank-overall', 'pts-scored', 'pts-allowed', 'conf-pts-scored', 'conf-pts-allowed']]))
        stats = pd.DataFrame(columns=pd.MultiIndex.from_tuples(stat_cols))
        stats.index.name = 'team'

        for k,team in self.teams_.items():
            stats.loc[k, ['name', 'div', 'conf']] = (team['name'], team['div'], team['conf'])
            stats.loc[k, ['overall','division','conference', 'vic_stren', 'sch_stren', 'misc']] = 0

        # special dataframe of game info to streamline processing. Contains 2 rows per game for
        # outcomes from each team's perspective. Perhaps someday this would be useful elsewhere
        reg_season = self.games_.xs('reg')
        games = pd.DataFrame(index=range(len(reg_season)*2), columns=['week','team','opp','wlt', 'scored', 'allowed'])
        z = 0
        for (_,game) in reg_season.iterrows():
            if game['p']:
                games.loc[z]   = [game['wk'], game['ht'], game['at'], NFL.result(game['hs'], game['as']), game['hs'], game['as']]
                games.loc[z+1] = [game['wk'], game['at'], game['ht'], NFL.result(game['as'], game['hs']), game['as'], game['hs']]
                z += 2

        games.dropna(inplace=True)

        for k,row in games.iterrows():
            stats.loc[row.team][('overall',row.wlt)] += 1
            stats.loc[row.team][('misc','pts-scored')] += row.scored
            stats.loc[row.team][('misc','pts-allowed')] += row.allowed
            if self.teams_[row.team]['div'] == self.teams_[row.opp]['div']:
                stats.loc[row.team][('division',row.wlt)] += 1

            if self.teams_[row.team]['conf'] == self.teams_[row.opp]['conf']:
                stats.loc[row.team][('conference',row.wlt)] += 1
                stats.loc[row.team][('misc','conf-pts-scored')] += row.scored
                stats.loc[row.team][('misc','conf-pts-allowed')] += row.allowed

        # strength of victory/schedule
        for (k,row) in stats.iterrows():
            # calculate strength of schedule and copy
            t = games[games.team==k]['opp']
            stats.loc[k]['sch_stren'] = stats.loc[t]['overall'].sum()

            # same for strength of victory
            t = games[(games.team==k) & (games.wlt=='win')]['opp']
            stats.loc[k]['vic_stren'] = stats.loc[t]['overall'].sum()

        # temporary table for calculating ranks - easier syntax
        t = pd.concat([stats['conf'], stats['misc'][['pts-scored','pts-allowed']]], axis=1)
        stats[('misc','rank-overall')] = (t['pts-scored'].rank() + t['pts-allowed'].rank(ascending=False)).rank()

        t['conf-off-rank'] = t.groupby('conf')['pts-scored'].rank()
        t['conf-def-rank'] = t.groupby('conf')['pts-allowed'].rank(ascending=False)
        t['conf-rank'] = t['conf-off-rank'] + t['conf-def-rank']
        stats[('misc', 'rank-conf')] = t.groupby('conf')['conf-rank'].rank()

        for i in ['overall','division','conference', 'sch_stren', 'vic_stren']:
            stats[(i,'pct')] = (stats[(i,'win')] + stats[(i,'tie')]*0.5) / stats[i].sum(axis=1).replace({0: np.nan})

        # sort by division tiebreakers
        self.stats = stats           # so that tiebreaks doesn't go recursive

        s = pd.Series(index=stats.index)
        for div in stats['div'].unique():
            z = self.tiebreaks(div)
            z[:] = range(len(z))
            s.loc[z.index] = z.astype(s.dtype)

        self.stats = stats.assign(divrank=s).sort_values(['div','divrank']).drop('divrank', level=0, axis=1)
        return self.stats

    def stash(self):
        '''Return a copy holding stashed game and week data, to be retored later
        '''

        if self.games_ is None:
            raise RuntimeError('game data has not yet been updated or loaded')

        return NFLState(self)


    def restore(self, stash):
        ''' Restores from the previous stash

        stash:  the object to restore from
        '''

        self.games_ = stash.games.copy()
        self.week   = stash.week
        self.stats  = None

    def update(self, season=None):
        ''' Updates team and game data from the underlying API

            season:  load only the specified season, else load the entire season (including pre- and post-game)
                     
                     This was originally conceived as a way to limit load times, but in practice it typically
                     doesn't save much, and it's limited to loading just the regular season since so much
                     of the module is dependent on that.
        '''

        if season and season != 'reg':
            raise NotImplementedError('Exclusion of regular season from update operations is not supported')

        self.teams_ = {}
        self.divs_  = {}
        self.confs_ = {}
        self.games_ = None

        for elem in self.engine.teams(self):
            key = elem['key']
            team = {
                'name': elem['fullname'],
                'short': elem['name'],
                'conf': elem['conf'],
                'div':  '-'.join([elem['conf'], elem['div']])
            }
 
            self.teams_[key] = team

        # engine returns a flat dataframe
        g = self.engine.games(self, season)
        g['p'] = (g['hs'].isna()==False) & (g['as'].isna()==False)

        # convert nans to 0's so we can set column type to int
        g[['as','hs']] = g[['as','hs']].fillna(0).astype('int64')
        self.games_ = g.sort_values('ts').set_index(['seas', 'id'])
        self._post_load()
        return self

    def _post_load(self):
        '''Perform cleanup operations after an update or load
        '''

        # divs_ and confs_ are dicts contains sets of team codes for each conference
        # and division. This are defined based on teams
        self.divs_ = {}
        self.confs_ = {}
        for (key,team) in self.teams_.items():
            (div,conf) = (team['div'],team['conf'])
            if self.divs_.get(div) is None:
                self.divs_[div] = set()

            if self.confs_.get(conf) is None:
                self.confs_[conf] = set()

            self.divs_[div].add(key)
            self.confs_[conf].add(key)

        # seasons_ is a dict where values are sets of team codes for each season, internally used
        # post will be undefined until regular season concludes
        self.seasons_ = {}
        teams = set(self.teams_.keys())
        for s in self.games_.index.get_level_values(0).unique():
            df = self.games_.xs(s)
            t = set(df['at'].unique()) | set(df['ht'].unique())
            self.seasons_[s] = t & teams

        if not self.week and 'reg' in self.seasons_:
            # assumes games are sorted chronologically
            now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            weekends = {}
            for (_,row) in self.games_.xs('reg').iterrows():
                t = row['ts'].replace(hour=0, minute=0, second=0, microsecond=0)
                weekends[row['wk']] = max(weekends.get(row['wk'], datetime.min), t)
                if t <= now:
                    self.week = row['wk']

            if self.week and self.week < max(self.weeks('reg')) and now > weekends[self.week]:
                # if between weeks, go to the next week
                self.week += 1

        if self.season not in self.seasons_:
            self.season = list(self.seasons_.keys())[0]

        self.stats = None

    @property
    def scoreboard(self):
        ''' Returns scoreboard display for the current week

            By default, this property will update the game database with final scores.
            Set autoUpdate=False to disable this behavior
        '''

        z = self.engine.scoreboard(self)
        if z and self.autoUpdate and self.year == z.year and z.season:
            for (k,v) in z.scoreboard.iterrows():
                if v['status'] == 'Final' or v['status'] == 'Final/OT':
                    self.set(z.week, season=z.season, **{v['hteam']: v['hscore'], v['ateam']: v['ascore']})

        return z

    @property
    def thisweek(self):
        ''' Returns a  display for the current week. This resembles the scoreboard
            but with less real-time information and more seasonal context.


            Unlike scoreboard, this property does *not* update the game database with final
            scores. It will update, however, if other operations update the database. For example,
            If you access the scoreboard property then the team records in subsequent gameday
            displays may also update to include current results. If you don't want this, you
            can do something like this:

            stash = nfl.stash()      # save game state (optional)
            nfl.clear(nfl.week)      # erase scores for the current week
            nfl.thisweek             # wlt, rank, streak etc now show status going *into* current week
            nfl.autoUpdate=False     # optional: prevent future dashboards from updating the database
            nfl.restore(stash)       # restore game state (optional)

        '''

        def fstreak(n):
            if n == 0: return '   '

            return '{:+3}'.format(n)

        def divrank(row):
            r = int(row[('misc','rank')])

            if r % 10 == 1: e = 'st'
            elif r % 10 == 2: e = 'nd'
            elif r % 10 == 3: e = 'rd'
            else: e = 'th'

            (conf, div) = row[('div','')].split('-')

            return '{}{} {}-{}'.format(r, e, conf, div[0])

        # gameday has the same basic structure as the scoreboard
        # we get the scoreboard directly from the engine to prevent autoUpdate
        src = self.engine.scoreboard(self)
        idx=pd.MultiIndex.from_product([['team','wlt','streak','rank'], ['away','home']])
        idx = idx.append(pd.MultiIndex.from_product([['misc'],['match','status','broadcast']]))
        gd = NFLDataFrame(columns=idx)
        gd.host = self
        standings = self.standings.copy()
        dates = self.schedule(weeks=src.week, by='game', ts=True)['date']
        streak = self.streak()
        for elem in standings['div'].unique():
            idx = standings[standings['div']==elem].index
            standings.loc[standings['div']==elem, ('misc','rank')] = pd.Series(range(1,len(idx)+1), index=idx)

        wlt = standings.xs('overall',axis=1,level=0).drop(columns='pct')
        orders = {'div': 1, 'conf': 2, 'league': 3}
        confpre = {'NFC': 0, 'AFC': 20}
        for k,row in src.scoreboard.iterrows():
            # key = '{}-{}'.format(row['ateam'], row['hteam'])
            key = k
            for (pos,t) in [('away',row['ateam']),('home',row['hteam'])]:
                gd.loc[key, ('team',pos)] = t
                gd.loc[key, ('wlt',pos)] = '-'.join(wlt.loc[t].astype(str).tolist())
                gd.loc[key, ('rank',pos)] = divrank(standings.loc[t])
                gd.loc[key, ('streak', pos)] = fstreak(streak[t])

            if row['state'] == 'pre':
                gd.loc[key, ('misc', 'status')] = dates[row['hteam']].strftime('%m/%d %H:%M')
            elif row['state'] == 'post':
                gd.loc[key, ('misc','status')] = 'F {:2}-{:2}'.format(row['ascore'], row['hscore'])
            else:
                gd.loc[key, ('misc','status')] = '  {:2}-{:2}'.format(row['ascore'], row['hscore'])

            gd.loc[key, ('misc', 'broadcast')] = row['broadcast']
            hdiv = standings['div'][row['hteam']]
            adiv = standings['div'][row['ateam']]
            if adiv == hdiv:
                gd.loc[key, ('misc','match')] = 'div'
            elif adiv.split('-')[0] == hdiv.split('-')[0]:
                gd.loc[key, ('misc', 'match')] = 'conf'
            else:
                gd.loc[key, ('misc','match')] = 'league'

        return gd

    @property
    def standings(self):

        return self._stats()[['name','div','overall','division','conference']]

    @property
    def NFC(self):

        return NFLConference('NFC', self)

    @property
    def AFC(self):

        return NFLConference('AFC', self)

    def set(self, ref, value=None, overwrite=True, ordered=False, season=None, **kwargs):
        ''' Set the final score(s) for games and weeks. You can use this to create
            hypothetical outcomes and analyze the effect on team rankings. Scores
            are specified by team code and applied to the specified week's schedule.
            If the score is specified for only one team in a game, the score of the other
            team is assumed to be zero if not previously set.

            ref:        should be one of:
                          an integer or list-like, specifying week numbers, with outcomes
                          specified in **kwargs or value

                          a dict, keyed by week numbers, containing dicts with team:outcome
                          pairings

                          a Pandas Series with a MultiIndex (index.name=['week','team]) outcomes

                          a NFLScenario

            value:      specify team codes and outcomes in lieu of kwargs. This can be an
                        NFLTeamMatrix, a Series (indexed by team codes) or a dict. In the
                        case of NFLTeamMatrix you can use kwargs to specify outcomes
                        not defined in the matrix. In other cases, kwargs are ignored

            overwrite:  overwrite previously set outcomes

            ordered:    **kwargs specify outcomes only (not scores), and in priority order.
                        This also applies if ref or value is set to a Series or dict

            **kwargs    dict of team codes and outcomes

            Values in **kwargs or ref (if dict or Series) can be:

                integers, taken as each team's final score, or

                'win', 'loss' or 'tie' - mapped to 1, 0 and -1 respectively

            You can explicitly set scores for both teams, but in many cases this is not necessary.
            The function will set the missing team's score to a logical value for the outcome
            you specify. For example, if one team's value is positive, and the other is missing
            that is interpreted as a win (or a spread), and the missing team's score is set to 0.
            Conversely, specifying 0 for only one team implies the other team wins and its score
            is set to 1. Specifying -1 for only one team implies a 0-0 tie. Explicitly setting
            incompatible values for both teams will yield illogical but consistent results
            (e.g. win/win actually results in a tie)

            # typical examples

            >>> nfl.set(7, MIN=17, GB=10)

            The above will set the score for the MIN/GB game in week 7 if
            MIN and GB play each other in that week. Otherwise, it will
            set each team's score respectively and default their opponent's scores
            to 0. Scores for bye teams in that week are ignored.

            >>> nfl.set(-1, PHI=1)
            >>> nfl.set(-1, PHI='win')  # same thing

            set PHI to win its remaining games

            >>> nfl.set(-1, PHI=0)
            >>> nfl.set(-1, overwrite=False, WSH=0)

            Set PHI to lose its remaining games, and WSH to lose its remaining games, except
            against PHI. In that case, the first call would set WSH
            to win the head-to-head match, and the second call would ignore the WSH/PHI game
            because it had been previously set

            >>> nfl.set(-1, ordered=True, DAL=0, WSH=1, PHI=0, NYG=1)

            Use the following hierarchy to set outcomes for the rest of the season:
              1) DAL loses
              2) WSH wins
              3) PHI loses (except against DAL)
              4) NYG wins (except against WSH)

            >>> s = pd.Series([1, 0, 0, 1], index=pd.MultiIndex.from_tuples([(17,'DAL'),(17,'PHI'),(18,'NYG'),(18,'WSH')]))
            >>> s
            17  DAL    1
                PHI    0
            18  NYG    0
                WSH    1
            dtype: int64

            >>> nfl.set(s, overwrite=False)

            set values from a Series, for games that haven't been played yet
        '''

        # First argument can also be a series of scores or outcomes. Index must be a MultiIndex (week,team)
        if isinstance(ref, NFLScenario):
            c = ['as', 'hs', 'p']
            for k,v in ref.items():
                if v == 'win':
                    self.games_.loc[k, c] = [0, 1, True]
                elif v == 'loss':
                    self.games_.loc[k, c] = [1, 0, True]
                elif v == 'tie':
                    self.games_.loc[k, c] = [0, 0, True]

            self.stats = None
            return

        elif isinstance(ref, pd.Series):
            for w in ref.index.get_level_values('week').unique():
                if type(w) is int:
                    self.set(w, overwrite, ordered, **ref.xs(w, level='week').to_dict())

            return

        elif isinstance(ref, dict):
            for k,v in ref.items():
                self.set(k, overwrite, ordered, **v)

            return

        elif isinstance(value, pd.Series):
            self.set(w, overwrite, ordered, **value.to_dict())
            return
        elif isinstance(value, dict):
            self.set(w, overwrite, ordered, **value)
            return

        season = season or self.season
        wk = self._weeks(ref)
        if isinstance(value, NFLTeamMatrix):
            teams = value.teams | set(kwargs.keys())
        else:
            teams = kwargs.keys()

        gm = self.games_.xs(season)
        idx = gm['wk'].isin(wk) & (gm['ht'].isin(teams) | gm['at'].isin(teams))
        if not overwrite:
            idx &= gm['p']==False

        idx = gm[idx].index

        for gid in idx:
            k = (season,gid)
            elem = self.games_.loc[k]
            (ht,at) = (elem['ht'],elem['at'])
            if isinstance(value, NFLTeamMatrix):
                t = value.score(ht,at)
                if len(t) == 2:
                    self.games_.loc[k, ['hs', 'as', 'p']] = [t[0], t[1], True]
                    continue

            if ordered:
                for t,v in kwargs.items():
                    v = ivmap(v)
                    if t == ht:
                        self.games_.loc[k, ['hs', 'as', 'p']] = [max(v,0), 1 if v==0 else 0, True]
                        break
                    elif t == at:
                        self.games_.loc[k, ['as', 'hs', 'p']] = [max(v,0), 1 if v==0 else 0, True]
                        break

            elif ht in teams and at in teams:
                self.games_.loc[k, ['hs', 'as', 'p']] = [max(ivmap(kwargs[ht]),0), max(ivmap(kwargs[at]),0), True]

            elif ht in teams:
                v = ivmap(kwargs[ht])
                hscore = max(v,0)
                if v < 0:
                    ascore = hscore
                else:
                    ascore = 0 if hscore > 0 else 1

                self.games_.loc[k, ['hs', 'as', 'p']] = [hscore, ascore, True]

            elif at in teams:
                v = ivmap(kwargs[at])
                ascore = max(v,0)
                if v < 0:
                    hscore = ascore
                else:
                    hscore = 0 if ascore > 0 else 1

                self.games_.loc[k, ['hs', 'as', 'p']] = [hscore, ascore, True]

        self.stats = None # signal to rebuild stats

    def clear(self, week, teams=None, season=None):
        '''Clear scores for a given week or weeks

        week:   can be an integer, range or list-like. Pass None to clear all (for whatever reason)

        teams:  limit operation to games for specified teams (list-like)
        '''

        season = season or self.season
        z = self.gameFrame(teams, week, season=season)[['p']]
        z.loc[:] = False
        self.games_.update(pd.concat({season: z}, names=['season']))
        self.stats = None


    def gameFrame(self, teams=None, weeks=None, allGames=False, season=None):
        '''Returns raw game data as a DataFrame. This is mostly used internally.
        '''

        if self.games_ is None:
            raise RuntimeError('game data has not been updated or loaded. Call update or load')

        season = season or self.season
        weeks = self._weeks(weeks)

        z = self.games_.xs(season)
        if weeks:
            z = z[z['wk'].isin(weeks)]

        if teams is not None:
            teams = self._list(teams)
            z = z[z['ht'].isin(teams) | z['at'].isin(teams)]

        if not allGames:
            z = z[z['p']]

        return z

    def games(self, teams=None, weeks=None, allGames=False, season=None):
        '''Iterate over the game database, returning a series
        '''

        for (_,game) in self.gameFrame(teams, weeks, allGames, season).iterrows():
            if game['p'] == False:
                game['as'] = game['hs'] = np.nan

            yield game

    def game(self, team, week, season=None):
        ''' return a single game for the given team and week 
        '''

        for game in self.games(team, week, allGames=True, season=season):
            return game

    def weeks(self, season=None):
        '''Returns a list of unique weeks for the specified season
        '''
        season = season or self.season
        return list(self.games_.xs(season)['wk'].unique())

    def schedule(self, teams=None, weeks=None, by='game', season=None, ts=False):
        ''' Return schedule for a week or team. If a single week and
        team are requested the result is a Series. If multiple weeks/teams
        are requested the result is a DataFrame, Multiple weeks and teams
        together will return a multi-indexed DataFrame.

        teams:      teams to return. This can be:
                       A single team code, e.g., 'MIN'
                       A list-like of team codes: ['MIN', 'PHI']
                       A division name: 'NFC-North''
                       A conference: 'NFC'

        weeks:      week to return or list-like of week numbers
                    Pass None for all weeks. Pass -1 for the remaining season

        by:         Data orientation, either 'team' or 'game'
                        'game' will return unique games based on the team(s) you request.
                        If two of the teams you request play each other the result will
                        be one row not two

                        'team' will return teams as rows; if two requested teams play
                        each other there will be two rows not one, with inverse 'home'
                        and 'away' designations. Additionally, 'bye' teams will be listed as empty rows

        season:     Override self.season

        ts:         Return dates as Pandas datetime objects that can be used computationally,
                    otherwise (default) return an easily read string

        As a convenience, you can also pass an integer as the first argument to get the schedule
        for the specified week. The following two calls are the same:

        schedule(5)
        schedule(teams=None, weeks=5)

        Examples
          nfl.schedule('MIN', by='team')   # Vikings schedule
          nfl.schedule(weeks=range(3,8))   # schedule for weeks 3-7
          nfl.schedule(5)                  # shortcut for week 5: single ints are taken as a week number not a team
          nfl.schedule('MIN', -1)          # Vikings remaining schedule (current week on)

        '''

        def to_date(obj):
            obj = pd.to_datetime(obj)
            if ts:
                return obj

            return '{}/{} {:02d}:{:02d}'.format(obj.month, obj.day, obj.hour, obj.minute)

        season = season or self.season
        if type(teams) is int and weeks is None:
            # special sugar: interpret first argument as a single week
            weeks = teams
            teams = teams2 = None
        else:
            teams2 = self._list(teams)
            if isinstance(teams, str) and len(teams2) > 1:
                # This is so we differentiate between a division/conference name and a team code
                teams = teams2

        weeks2 = self._weeks(weeks)
        if isinstance(weeks, int) and weeks < 1:
            # similar: don't confuse -1 with an actual week number
            weeks = weeks2

        if by == 'game':
            games = NFLDataFrame(self.gameFrame(teams=teams2, weeks=weeks2, allGames=True, season=season).copy())
            games.host = self
            df = games.rename({ 'wk': 'week', 'ht': 'hteam', 'at': 'ateam', 'hs': 'hscore', 'as': 'ascore', 'ts': 'date'}, axis=1)

            if (df['p']==False).any():
                # this assignment will change dtype to float64, so only do it if necessary
                df.loc[df['p']==False, ['ascore', 'hscore']] = np.nan

            df = df[['week', 'hteam', 'ateam', 'hscore', 'ascore', 'date']]
            if not ts:
                df['date'] = df['date'].dt.strftime('%m/%d %H:%M')

            if isinstance(teams, (str, NFLTeam)) and type(weeks) is int:
                # need a special test here to make sure we have a data frame
                if len(df) > 0:
                    return df.drop('week', axis=1).iloc[0]
                else:
                    # return an empty series
                    return pd.Series(index=df.columns.drop('week'))

            elif type(weeks) is int:
                return df.set_index('hteam').drop('week', axis=1)

            return df.set_index(['week', 'hteam'])

        if teams2 is None:
            teams2 = list(self.seasons_[season])
            teams2.sort()
        elif set(teams2) - self.seasons_[season]:
            # this might occur if they select teams that aren't in the playoffs
            teams2 = list(set(teams2) & self.seasons_[season])
            teams2.sort()

        # Here we construct the database index in advance so that it includes empty
        # rows for teams with bye weeks
        if weeks2 is None:
            weeks2 = self.weeks(season)
        else:
            w2 = self.weeks(season)
            weeks2 = list(filter(lambda x: x in w2, weeks2))

        df = NFLDataFrame(index=pd.MultiIndex.from_product([weeks2, teams2], names=['week', 'team']),
                columns=['opp', 'loc', 'score', 'opp_score', 'wlt', 'date'])
        df.host = self

        for game in self.games(teams=teams2, weeks=weeks2, allGames=True, season=season):
            if game['ht'] in teams2:
                df.loc[(game['wk'],game['ht'])] = [game['at'], 'home', game['hs'], game['as'], NFL.result(game['hs'], game['as']), game['ts']]
            
            if game['at'] in teams2:
                df.loc[(game['wk'],game['at'])] = [game['ht'], 'away', game['as'], game['hs'], NFL.result(game['as'], game['hs']), game['ts']]

        df['date'] = df['date'].astype('datetime64[ns]')
        if not ts:
            df['date'] = df['date'].dt.strftime('%m/%d %H:%M')

        if isinstance(teams, str) and type(weeks) is int:
            return df.loc[(weeks,teams)]
        elif isinstance(teams, str):
            return df.xs(teams, level=1)
        elif isinstance(teams, NFLTeam):
            return df.xs(teams.code, level=1)
        elif type(weeks) is int:
            return df.xs(weeks, level=0)

        return df


    def opponents(self, teams, weeks=None, allGames=False, season=None):
        ''' Returns the set of common opponents of one or more teams

            The teams argument can be a single team or a list.
        '''

        if teams is None:
            raise ValueError("teams cannot be None here")

        teams = self._list(teams)

        ops = {t:set() for t in teams}

        for game in self.games(teams, weeks=weeks, allGames=allGames, season=season):
            if game['ht'] in ops:
                ops[game['ht']].add(game['at'])

            if game['at'] in ops:
                ops[game['at']].add(game['ht'])

        # Resulting set is common (intersection) of opponents excluding the teams themselves
        z = None
        for s in ops.values():
            if z is None:
                z = s
            else:
                z &= s

        z -= set(teams)
        return z

    def roster(self, team):
        '''Return team roster

        team: team code
        '''

        if team not in self.rosters_:
            self.rosters_[team] = self.engine.roster(self, team)

        return self.rosters_[team]

    def boxscore(self, team, week=None, season=None):

        if week is None:
            game = self(team).active_game
        else:
            game = self.game(team, week, season)

        return self.engine.boxscore(self, game)


    def streak(self, teams=None, weeks=None):
        '''Returns the streak for the specified teams, either as a number (if a single team is passed)
           or as a Series (if a list-like is passed)
        '''

        teams2 = self._list(teams)

        sched = self.schedule(teams=teams2, weeks=weeks, by='team').dropna().reset_index().set_index(['team', 'week']).sort_index()['wlt']
        s = pd.Series(0, index=sched.index.unique(0), dtype=int)
        for t in s.index:
            z = sched.xs(t)
            if len(z) > 0 and z.iloc[-1] != 'tie':
                n = 0

                while n < len(z) and z.iloc[-n-1] == z.iloc[-1]:
                    n += 1

                if z.iloc[-1] == 'loss':
                    n *= -1

                s[t] = n


        if isinstance(teams, str) and len(teams2) == 1:
            # return the value for a single team
            return s[teams]

        return s


    def player_stats(self, id, keys=False, stack=None):
        '''Return statistics for the specified player.
           For now, unlike rosters, statistics are not cached

           id:  if a scalar, this is interpreted as a engine-specific player identifier
                if a list-like you can pass id's or NFLPlayer objects to return a multi-index dataframe

           stack: if returning multiple players, specify the multi-index key. Typical values are 'id' and 'name'.
                  Note that ids are always used if ids are supplied as arguments
        '''


        if isinstance(id, NFLPlayer):
            stats = self.engine.player_stats(self, id['id'], keys=keys)
            return stats.xs('', axis=1)

        if type(id) is str:
            stats = self.engine.player_stats(self, id, keys=keys)
            return stats.xs('', axis=1)

        # else, iterate over list
        stats = None
        if not stack:
            stack = 'id'

        for elem in id:
            if isinstance(elem, NFLPlayer):
                df = self.engine.player_stats(self, elem['id'], keys=keys, label=elem[stack])
            else:
                 df = self.engine.player_stats(self, elem, keys=keys, label=elem)

            if stats is None:
                stats = df
            else:
                stats = pd.concat([stats, df], axis=1)

        return stats

    def net_stats(self, teams, type='Touchdown', season=None, wrapper=None, wrapper_args={}):
        '''Returns a dict of game statistics for the specified teams.
           The statistics are the sum of "net" game events; that is, events achieved by each
           team less those achieved by their opponent. This is primarily designed
           to compute "net touchdowns" for tiebreaker computations, but theoretically could
           be used for field goals, interceptions, safeties, etc
        '''

        # Some actions may be characterized in multiple ways, for instance, defensive touchdowns
        # The methods dict provides for this, and negative values count in reverse
        methods = {
          'Touchdown': {'Touchdown': 1, 'Interception Touchdown': -1}
        }

        teams = self._list(teams)
        season = season or self.season
        tds = dict.fromkeys(teams, 0)

        method = methods.get(type, {type: 1})

        # FIXME: implement a complete game generator in lieu of this kludge
        if wrapper:
            pb = wrapper(total=len(self.gameFrame(teams, season=season)), **wrapper_args)
        else:
            pb = None

        for game in self.games(teams, season=season):
            (ht,at) = (game['ht'],game['at'])
            drives = self.engine.drives(self, game)
            if pb:
                pb.update()

            for (k,v) in method.items():
                t = drives[drives['result']==k].reset_index().groupby('team').count()['pts']
                ht_td = (t.get(ht, 0) - t.get(at, 0)) * v
                if ht in teams:
                    tds[ht] += ht_td

                if at in teams:
                    tds[at] -= ht_td

        return tds

    def wlt(self, teams=None, within=None, weeks=None, allGames=False, season=None, result='short'):
        '''Return the wlt stats of one or more teams

        teams:  team code or list of team codes

        within: list of opponent team codes that defines the wlt universe

        weeks:  limit to specified weeks

        season: override default season

        allGames: return all games (includes unplayed games)

        result: the type of result needed. 'short' is by far the most common, but you can also specify
                'long' - include score columns
                'extended' - return a tuple, first element is the long-style wlt table, 2nd element
                is a team matrix of games played against each other
        '''

        teams  = self._list(teams)
        within = self._list(within)

        cols = ['win','loss','tie', 'pct', 'scored','allowed']

        df = pd.DataFrame(index=list(teams), columns=cols)
        df[df.columns] = 0
        df.columns.name = 'outcome'
        df.index.name = 'team'

        # define a matrix of games played against opponents
        m = NFLGameMatrix(teams)

        for game in self.games(teams, weeks=weeks, allGames=allGames, season=season):
            if game['ht'] in teams and (within is None or game['at'] in within):
                z = NFL.result(game['hs'], game['as'])
                if z:
                    df.loc[game['ht'], z] += 1
                    df.loc[game['ht'], 'scored'] += game['hs']
                    df.loc[game['ht'], 'allowed'] += game['as']

                if game['at'] in m.columns:
                    if z == 'win':
                        m.loc[game['ht'], game['at']] += 1
                    elif z == 'tie':
                        m.loc[game['ht'], game['at']] += 0.5

            if game['at'] in teams and (within is None or game['ht'] in within):
                z = NFL.result(game['as'], game['hs'])
                if z:
                    df.loc[game['at'], z] += 1
                    df.loc[game['at'], 'scored'] += game['as']
                    df.loc[game['at'], 'allowed'] += game['hs']


                if game['ht'] in m.columns:
                    if z == 'win':
                        m.loc[game['at'], game['ht']] += 1
                    elif z == 'tie':
                        m.loc[game['at'], game['ht']] += 0.5


        df['pct'] = (df['win'] + df['tie'] * 0.5) / df.drop(columns=['scored','allowed'], errors='ignore').sum(axis=1)
        df.sort_values('pct', ascending=False, inplace=True)
        
        if result == 'long':
            return df
        elif result == 'extended':
            return (df, m)

        return df.drop(columns=['scored','allowed'])

    def team_stats(self, team):
        '''Return stats for a single team
        '''

        return self._stats().loc[team][['overall','division','conference']].unstack()


    def tiebreakers(self, teams):
        '''Return tiebreaker analysis for specified teams

        Each row in the returned dataframe is the results of a step in the NFL's tiebreaker procedure
        currently defined here: https://www.nfl.com/standings/tie-breaking-procedures

        Rows are in order of precedence and depend on whether the teams are in the same division or not

        rank (conference or overall) statistics are always in increasing order, e.g. 1 is the worst
        ranked team

        An occurrence of 'inf' as a value indicates that comparisons are not valid for that rule
        in the given context as per the tiebreakers procedures. The win/loss/tie statistics are still
        provided for information. For example, if the specified teams do not all have the same number
        of head-to-head matchups, or a wild-card tiebreaker does not have at least 4 common games, the
        'pct' column is set to 'inf' for those rules.

        Example:

        z = nfl.tiebreakers(nfl('NFC-North').teams)

        # sort division according to tiebreaker rules, highest ranked team in column 1
        z = z.xs('pct', level=1, axis=1).sort_values(list(z.index), axis=1, ascending=False)
        '''

        teams = self._list(teams)
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([teams, ['win','loss','tie', 'pct']], names=['team','outcome']))
        common_opponents = self.opponents(teams)
        divisions = set()
        stats = self._stats()
        if self.netTouchdowns:
            ntd = self.net_stats(teams)
        else:
            ntd = {}

        # determine which divisions are in the specified list. If more than one then adjust the tiebreaker order
        for t in teams:
            divisions.add(self.teams_[t]['div'])

        # set rules to default values here so they appear in the correct order
        df.loc['overall'] = np.nan
        df.loc['head-to-head'] = np.nan
        if len(divisions) > 1:
            df.loc['conference'] = np.nan
            df.loc['common-games'] = np.nan
        else:
            df.loc['division'] = np.nan
            df.loc['common-games'] = np.nan
            df.loc['conference'] = np.nan
        
        df.loc['victory-strength'] = np.nan
        df.loc['schedule-strength'] = np.nan
        df.loc['conference-rank'] = np.nan
        df.loc['overall-rank'] = np.nan

        if len(divisions) > 1:
            df.loc['conference-netpoints'] = np.nan
        else:
           df.loc['common-netpoints'] = np.nan

        df.loc['overall-netpoints'] = np.nan

        if self.netTouchdowns:
            df.loc['net-touchdowns'] = np.nan

        (h2h,gm) = self.wlt(teams, within=teams, season='reg', result='extended')
        co  = self.wlt(teams, within=common_opponents, season='reg', result='long')

        for team in teams:
            df.loc['overall', team] = stats.loc[team,'overall'].values
            df.loc['head-to-head', team] = h2h.loc[team].drop(['scored', 'allowed']).values
            df.loc['common-games', team] = co.loc[team].drop(['scored','allowed']).values
            df.loc['conference', team] = stats.loc[team,'conference'].values
            df.loc['victory-strength', team] = stats.loc[team,'vic_stren'].values
            df.loc['schedule-strength', team] = stats.loc[team,'sch_stren'].values
            if 'division' in df.index:
                df.loc['division', team] = stats.loc[team,'division'].values

            df.loc['conference-rank', (team,'pct')] = stats.loc[team, ('misc', 'rank-conf')]
            df.loc['overall-rank', (team,'pct')] = stats.loc[team, ('misc', 'rank-overall')]
            if 'common-netpoints' in df.index:
              df.loc['common-netpoints', (team,'pct')] = co.loc[team, 'scored'] - co.loc[team, 'allowed']

            if 'conference-netpoints' in df.index:
                df.loc['conference-netpoints', (team,'pct')] = stats.loc[team, ('misc', 'conf-pts-scored')] - stats.loc[team, ('misc', 'conf-pts-allowed')]

            df.loc['overall-netpoints', (team,'pct')] = stats.loc[team, ('misc', 'pts-scored')] - stats.loc[team, ('misc', 'pts-allowed')]

            if self.netTouchdowns:
                df.loc['net-touchdowns', (team,'pct')] = ntd.get(team, np.nan)

            # sanity checks: we use inf to indicate that the column should be ignored
            if not gm.same():
                # all teams must have played each other the same number of games or h2h is invalid
                df.loc['head-to-head', (team,'pct')] = np.inf


        # team-wide sanity checks
        if (df.loc['common-games'].drop('pct', level=1).groupby('team').sum() < 4).any():
            # if any team plays less than 4 common games, all common-team record scores are invalid
            df.loc['common-games'].loc[:, 'pct'] = np.inf

        # A note about how 'clean-sweep' analysis works
        # The sanity check below ensures that at least one team is a "clean-sweep" either perfectly winning
        # or losing against the others. If there is no "clean-sweep" (pct=[0,1]) then the rule is skipped
        #   If there is a clean-sweep loser (pct=0) that team will be dropped from contention by sorting, and the
        #   procedure restarted with the remaining teams
        #   If there is a clean-sweep winner, then the lower-ranked teams will still be dropped one-by-one
        #   as the procedure restarts i.e. a clean-sweep winner in a set of opponents is by definition
        #   a clean-sweep winner of a subset of opponents

        z = df.loc['head-to-head'].loc[:,'pct']
        if len(divisions) > 1 and len(z) > 2 and not z.isin([0, 1]).any():
            # wildcard tiebreakers with 3+ teams must be a clean sweep. If there isn't one then set to nan
            df.loc['head-to-head'].loc[:, 'pct'] = np.inf

        return df


    def tiebreaks(self, teams, limit=None, divRule=True):
        '''Returns a series with the results of a robust tiebreaker analysis for teams
           Team codes comprise the series index in hierarchical order, while the series
           value indicates the rule (or basis) for each team besting the one below it

           teams     a list-like of team codes, or a division/conference code

           limit     return once this number of tiebreakers has been determined

           divRule   Enforce the "one-club-per-division" rule as stated in the wildcard
                     tiebreaking procedures
        '''
        teams = list(self._list(teams))
        limit = limit or len(teams)
        r = pd.Series(name='level')

        # shortcuts for efficiency: implement before call to _stats for speed
        if len(teams) == 0:
            return r
        elif len(teams) == 1:
            r[teams[0]] = 'overall'
            return r
        elif limit == 1:
            # if only care about the winner, try to first discern based on overall pct
            if self.stats is None:
                z = self.wlt(teams, season='reg')['pct']
            else:
                z = self.stats.loc[teams,('overall','pct')]
                
            z = z.copy().sort_values(ascending=False)
            if z.diff(-1).iloc[0] > 0:
                r[z.index[0]] = 'overall'
                return r

        stats = self._stats()

        def check_until_same(s):
            '''Returns the number of initial elements with the same value as the next one.
               0 means the first element is unique
               len(s)-1 means they are all the same
            '''

            if s.isna().all() or (s == np.inf).any():
                return len(s) - 1
                
            i = 0
            for (k,v) in s.diff(-1).items():
                if v != 0:
                    return i

                i += 1

            return len(s)-1

        def msg(op, codes):
            return 'tiebreaks: {} - {}'.format(op, ','.join(codes))

        def test(t):
            '''Runs tests to determine the exclusive highest-ranking team.
               If successful, a tuple (team_code,rule) is returned
               If a condition is encountered requiring a restart on a subset
               of teams, a tuple (surviving_teams, rule) is returned

               t    the set of teams to test. Per wildcard rules, only 1 team from each division
                    should be included (it does not test for this)
            '''

            # count divs
            divs = set()
            for elem in t:
                divs.add(self.teams_[elem]['div'])

            z = self.tiebreakers(t).xs('pct', level=1, axis=1).T

            # sort by values for each rule, and drop the 1st rule (overall) since that was already
            # tested by the calling function
            z = z.sort_values(list(z.columns), ascending=False).drop(z.columns[0], axis=1)
            for rule in z.columns:
                if rule == 'head-to-head' and len(divs) > 1 and len(z) > 2:
                    # special case for ties amongst 3+ teams across divisions
                    # a "clean sweep" either picks the top team or eliminates the bottom one
                    # Note the tiebreakers function ensures that if there isn't a clean winning or losing sweep
                    # then the rule is invalidated
                    if z.iloc[0][rule] == 1:
                        return (z.index[0], rule)
                    elif z.iloc[-1][rule] == 0:
                        return (t - {z.index[-1]}, rule)
                else:
                    x = check_until_same(z[rule])
                    if x == 0:
                        return (z.index[0], rule)
                    elif x < len(z)-1:
                        # multiple top teams at this rule, so eliminate the remainder for this round
                        # and start over
                        return (t - set(z.index[x+1:]), rule)

            # Early in the season it's possible that some teams haven't played; in that case
            # just return the first team
            if self.stats.loc[list(t),'overall'].sum().sum() == 0:
                return (list(t)[0], '')

            suggest = '' if self.netTouchdowns else ' (try setting nfl.netTouchdowns=True)'
            raise RuntimeError("Can't resolve tiebreaker: teams are essentially equal.{} ({})".format(suggest, ','.join(t)))

        while len(teams) > 1:
            if len(r) >= limit:
                return r

            logging.debug(msg('1.1 (Begin)', teams))
            subTeams = set(teams)

            # apply division tiebreakers if necessary. start by counting # of divisions
            z = stats.loc[teams] # NB: this will change row order
            divs = z['div'].value_counts()

            # check if a) divRule in effect, b) are 2+ divisions, c) are 1+ divisions with 2+ clubs
            if divRule and len(divs) > 1 and len(divs[divs>1]) > 0:
                for d in divs[divs>1].index:
                    # remove teams in this division except for the first
                    subTeams -= set(stats[stats.index.isin(z[z['div']==d].index)].index[1:])

                logging.debug(msg('2.2 (1 club/division rule)', subTeams))

            # we first test the overall rule since that can be done inexpensively and hopefully
            # will eliminate a significant number of candidates. Otherwise,
            # we call the test function which calculates all the tiebreaker rules (at a significant
            # expense), which we call with the surviving subset of teams
            z = stats.loc[list(subTeams),('overall','pct')].sort_values(ascending=False).copy()
            t = check_until_same(z)
            if t == 0:
                r[z.index[0]] = 'overall'
                teams.remove(z.index[0])
                logging.debug(msg('3.1 (Select)', [z.index[0],'overall']))
            else:
                subTeams = set(z.index[:t+1])
                while True:
                    logging.debug(msg('1.2 (Test)', subTeams))
                    (result,rule) = test(subTeams)
                    if type(result) is set:
                        logging.debug(msg('1.3 (Restart)', list(result) + [rule]))
                        subTeams = result
                    else:
                        r[result] = rule
                        teams.remove(result)
                        logging.debug(msg('3.2 (Select)', [result,rule]))
                        break
    
        # should always be one team left (runt of the litter)
        r[teams[0]] = ''
        return r

    def wildcard(self, teams, seeds=3):
        '''Run a wildcard analysis on the specified teams. This consists
           of running tiebreakers for all subsets of the specified teams.
           The resulting DataFrame includes each subset and their tiebreaker
           results.

           I'm not sure how useful this actually is, so it may be removed
           from a future update. The subsets subroutine may be useful on its
           own though.
        '''

        def subsets(data, size, unique=True):
            '''Via a generator, returns data subsets of the specified size

               data:   array of values
               size:   size of the subsets
               unique: specifies whether to iterate over previous values:

                       subsets([1, 2, 3], 2, False) returns:
                         [1, 2]
                         [1, 3]
                         [2, 3]

                       subsets([1, 2, 3], 2, True) returns:
                         [1, 2]
                         [1, 3]
                         [2, 1]
                         [2, 3]
                         [3, 1]
                         [3, 2]
            '''

            n = 0
            while n < len(data):
                p = data[n]
                n += 1
                if size == 1:
                    yield [p]
                elif unique:
                    for elem in subsets(data[n:], size-1):
                        yield [p] + elem    
                else:
                    for elem in subsets(data[:n-1] + data[n:], size-1):
                        yield [p] + elem

        index = []
        for i in range(2, len(teams)+1):
            for elem in subsets(list(teams), i):
                index.append(' '.join(elem))

        df = pd.DataFrame(index=index, columns=range(1, seeds+1))
        df.index.name = 'tiebreaker'
        df.columns.name = 'order'
        for i in df.index:
            tb = self.tiebreaks(i.split()).index[:seeds]
            df.loc[i] = ''
            df.loc[i, df.columns[:len(tb)]] = tb

        return df


    @staticmethod
    def result(a, b):
        
        if a is np.nan or b is np.nan:
            return ''
        elif a > b:
            return 'win'
        elif a < b:
            return 'loss'

        return 'tie'

    def _weeks(self, weeks):

        if type(weeks) is int:
            if weeks <= 0:
                return range(self.week,19)

            return [weeks]

        if weeks is None:
            return None

        if isinstance(weeks, pd.core.base.IndexOpsMixin):
            # just need to cast
            return list(weeks)

        if hasattr(weeks, '__iter__'):
            # assume no problems iterating
            return weeks

        # this shouldn't happen: maybe if they pass some sort of float?
        return [weeks]

    def _list(self, teams):
        '''Transforms objects into list-likes. This is designed so that
           the user can pass objects or team/division/conference codes
           as arguments to functions that expect lists of team codes or
           week numbers

           - None becomes None
           - NFLTeam, NFLDivision and NFLConfeence objects become team code lists
           - team/division/conference codes becomes team code lists
           - other iterables (of any type) are cast to ordinary lists
        '''

        if teams is None:
            return None

        if isinstance(teams, pd.core.base.IndexOpsMixin):
            # a list-like that needs to be cast
            return list(teams)

        if isinstance(teams, NFLTeam):
            return [teams.code]

        if isinstance(teams, (NFLDivision, NFLConference)):
            return list(teams.teams)

        if not isinstance(teams, str) and hasattr(teams, '__iter__'):
            # assume no problems iterating
            return teams

        # below here assume a scalar
        if teams in self.divs_:
            # division code
            return self.divs_[teams]

        if teams in self.confs_:
            # conference code
            return self.confs_[teams]

        # single team code or integer
        return [teams]

    def scenarios(self, teams, weeks, spots=1, ties=True, wrapper=None, wrapper_args={}):
        '''Returns a dataframe of scenarios and outcomes with the specified constraints

           This function is essentially a wrapper for NFLScenarioMaker with
           the most common use cases, i.e. determining playoff spots. You
           can customize and optimize the model by implementing NFLScenarioMaker
           directly (see docs).

           teams    list-like of teams for which to generate scenarios

           weeks    weeks for which to generate scenarios

           spots    Sets the number of eligible playoff spots for this analysis.
                    For example, if set to 1 then only the top team will
                    be playoff eligible for each scenario. This is how you
                    typically would analysis potential division championships,
                    passing a set of teams for a given division.

                    If set to 0 then NFLConference.playoffs() determines
                    eligibility. In this case, an additional sanity check
                    is made to ensure teams are all from the same conference.

           ties     whether to include ties as possible outcomes

           wrapper  A class used to wrap the scenario iterator. This is typically
                    used to implement progress bars. If the class includes an 'update'
                    function it is called after each scenario iteration, in the manner
                    of the tqdm package

           wrapper_args Additionaal arguments to instantiate the wrapper

           The resulting DataFrame will have 3**games rows, each row representing a unique
           scenario and outcome, where rows is the number of games played collectively by teams in the
           range of weeks. If ties=False the DataFrame will have 2**games rows.
           The DataFrame will have a MultiIndex column format, the product of weeks
           and teams, with an extra 1st level named 'playoffs' that includes Boolean columns
           for each team indicating whether they test positive (based on the value of spots)
           for that scenario.

           The example below runs all possible scenarios for weeks 17 and 18, testing whether
           NE or BUF have the superior record compared to each other. This can be used to
           determine which team might win the division championship (NE and BUF are both AFC-East
           teams), assuming that no other teams (i.e MIA and NYJ) are contenders.

           >>> nfl.scenarios(['NE', 'BUF'], [17,18])
           week    17         18        outcome       
           team    NE  BUF    NE   BUF       NE    BUF
           0     loss  win   win   win    False   True
           1     loss  win   win  loss     True  False
           . . .

           Examples:

           # determine which teams can still win their division after week15, not allowing for ties
           nfl.scenarios('NFC-North', range(16,19), ties=False)

           # determine playoff scenarios for an entire conference: this will take a *long* time
           # if obvious ineligibles are not first weeded out
           nfl.scenarios('AFC', [17, 18], spots=0)
        '''

        teams = self._list(teams)
        if spots == 0:
            # ascertain the conference with sanity check
            conf_teams = {}
            for k,v in self.confs_.items():
                for elem in v:
                    conf_teams[elem] = k

            c = set(conf_teams[k] for k in teams)
            if len(c) > 1:
                raise ValueError('Teams must all belong to the same conference')

            conf = NFLConference(list(c)[0], self)

        if wrapper is None:
            wrapper = NFLEmptyContextWrapper
        
        with NFLScenarioMaker(self, teams, weeks, ties) as gen, wrapper(gen, **wrapper_args) as wgen:
            df = gen.frame(['outcome'])
            for option in wgen:
                x = len(df)
                df.loc[x] = option.to_frame()
                df.loc[x, 'outcome'] = False
                self.set(option, season='reg')
                if spots == 0:
                    p = conf.playoffs()
                    z = [('outcome',i) for i in set(teams) & set(p.index)]
                    df.loc[x, z] = True
                else:
                    tb = self.tiebreaks(teams, limit=spots)
                    z = [('outcome',i) for i in tb.index[:spots]]
                    df.loc[x, z] = True

            return df

    @staticmethod
    def _engine(name):
        return getattr(sys.modules[__name__], name)()

        if name == 'ProFootballRef':
            return NFLSourceProFootballRef()
        elif name == 'ESPN':
            return NFLSourceESPN()

        raise ValueError('Unrecognized engine: {}'.format(name))

    @staticmethod
    def _name():
        return (__name__,globals())

class NFLScoreboard():
    '''A wrapper class for scoreboard display of live games. Typically you
       obtain or display this through the nfl.scoreboard propety.

       Class attributes:
            week        scoreboard week
            year        scoreboard year
            scoreboard  pandas DataFrame
    '''

    def __init__(self, nfl, year, week, season, scoreboard):
        self.nfl  = nfl
        self.year = year
        self.week = week
        self.season = season
        self.scoreboard = NFLDataFrame(scoreboard)
        self.scoreboard.host = nfl

    def __repr__(self):
        if isinstance(self.scoreboard, pd.DataFrame):
            return 'Week {}\n'.format(self.week) + self.scoreboard.drop('state',axis=1).__repr__()

        return ''

    def _repr_html_(self):
        if isinstance(self.scoreboard, pd.DataFrame):
            return '<h3>Week {}</h3>\n'.format(self.week) + self.scoreboard.drop('state',axis=1)._repr_html_()

        return ''

    def __call__(self, teams=None, state=None):
        '''Return scoreboard for just the specified team(s).
        '''

        z = self.scoreboard
        if teams:
            teams = self.nfl._list(teams)
            z = z[z['ateam'].isin(teams) | z['hteam'].isin(teams)]
        
        if state:
            z = z[z['state']==state]

        return z.drop('state',axis=1)

class NFLEmptyContextWrapper():
    ''' An empty wrapper to use in code requiring a context object when none
        is specified by the user
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

class NFLPlayDescWrapper(str):
    '''Just a bit of syntax sugar to print a clean string to the console - no need to wrap in a print() function
    '''
    def __repr__(self):
        '''Return unformatted string
        '''
        return self

class NFLPlay(pd.Series):

    @property
    def _constructor(self):
        return NFLPlay

    @property
    def _constructor_expanddim(self):
        return NFLPlaysFrame

    @property
    def text(self):
        '''Return the readout of a single play. Negative values are treated
           as offsets (i.e. from the end of the frame) while non-negative values
           are treated as ordinary keys
        '''
        return (NFLPlayDescWrapper(self['desc']))

class NFLDataFrame(pd.DataFrame):
    ''' Subclass of Pandas.DataFrame that displays recognized columns as
        team names and codes, in lieu of just codes.

        This works by overriding the __repr__ and _repr_html functions
        and returning a formatted copy of itself. To make this work you must
        also set the host property to an NFL object. The class then references
        the 'display' property on that object to determine how to display
        teams ('data' and 'user' are currently recognized values)
    '''

    _metadata = ['host']
    host = None

    def __repr__(self):

        if self.host and self.host.display == 'user':
            return self._reformat().__repr__()

        # else, default display
        return super().__repr__()

    def _repr_html(self):

        if self.host and self.host.display == 'user':
            return self._reformat()._repr_html_()

        # else, default display
        return super()._repr_html_()        

    def _reformat(self):
        
        def remap_val(s):
            t = self.host.teams_
            return s.map(lambda x: '{} {:>3}'.format(t.get(x,{}).get('short'), x) if x in t else '')

        def remap_idx(s):
            fd_size = max(map(lambda x: len(self.host.teams_[x]['short']), s))
            t = self.host.teams_
            return s.map(lambda x: '{} {:>3}'.format(t.get(x,{}).get('short').ljust(fd_size), x) if x in t else '')

        field_names = ['opp', 'team', 'hteam', 'ateam', ('team','away'), ('team','home')]

        z = pd.DataFrame(self)
        for c in z.columns:
            if c in field_names:
                z[c] = remap_val(z[c])

        if z.index.nlevels == 1 and z.index.name in field_names:
            z.set_index(pd.Index(remap_idx(z.index), name=z.index.name), inplace=True)
        elif z.index.nlevels > 1:
            for i in range(z.index.nlevels):
                if z.index.names[i] in field_names:
                    z.set_index(z.index.set_levels(remap_idx(z.index.levels[i]), level=i), inplace=True)

        return z

class NFLPlaysFrame(NFLDataFrame):

    _metadata = ['gameInfo', 'host']

    @property
    def _constructor(self):
        return NFLPlaysFrame

    @property
    def _constructor_sliced(self):
        return NFLPlay

    def __repr__(self):
        s = self.gameInfo
        if s:
            s += '\n'

        return s + super().__repr__()

    def text(self, offset=0):
        '''Returns the readout from the most recent play, counting from the bottom

           offset:      offset from the end of the DataFrame
        '''
        return self.iloc[-(offset+1)].text

class NFLPlayer(pd.Series):
    '''Encapsulates a player in the roster
    '''

    def __init__(self, series, host):
        super().__init__(series)
        self.host = host

    def __repr__(self):
        if self.get('jersey') is np.nan:
            return '{}: {}'.format(self['name'], self['position'])

        return '{}: {} ({})'.format(self['name'], self['position'], self['jersey'])

    @property
    def stats(self):
        ''' Synonym for player_stats()
        '''
        return self.host.player_stats(self)

    @property
    def games(self):
        '''return a series of regular-season games played
        '''

        return self.host.engine.games_per_player(self.host, self['id'])


#   def _repr_html_(self):


class NFLRoster():
    '''Contains a team roster
    '''

    def __init__(self, roster, host):
        self.roster = roster
        self.host = host

    def __getattr__(self, key):
        '''Returns the portion of the roster for the side with the specified property name
        Examples:
        min = nfl('MIN').roster
        min.SPEC
        '''

        df = self.roster[self.roster['side']==key]
        if len(df) == 0:
            sides = '{' + ' '.join(self.sides) + '}'
            raise AttributeError('property must be one of {}'.format(sides))

        return df

    def __getitem__(self, key):
        '''If an int is passed, returns the NFLPlayer with that jersey number.
        If a str is passed, returns the portion of the roster for the given position code
        '''

        if type(key) is int:
            df = self.roster[self.roster['jersey']==str(key)]
            if len(df) > 0:
                return NFLPlayer(df.iloc[0], self.host)

            return None

        df = self.roster[self.roster['pos']==key]
        if len(df) == 0:
            positions = '{' + ' '.join(self.positions) + '}'
            raise KeyError('key must be one of {}'.format(positions))

        return df

    @property
    def coach(self):
        '''Team's coach
        '''
        return self.member('COACH')

    @property
    def positions(self):
        '''List of position codes in the roster
        '''

        return set(self.roster['pos'].unique())

    @property
    def sides(self):
        '''List of side codes in the roster (i.e. valid properties)
        '''

        return set(self.roster['side'].unique())

    def name(self, t):
        '''Search roster by name or partial name
        '''
        return self.roster[self.roster['name'].str.contains(t, case=False)]

    def member(self, code):
        '''Name and jersey for the 1st roster member with the given code
        '''
        return NFLPlayer(self[code].iloc[0], self.host)
