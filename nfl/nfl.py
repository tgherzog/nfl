#!/usr/local/bin/python

import openpyxl
import sys
import logging
import copy
from docopt import docopt
from datetime import datetime
import numpy as np
import pandas as pd

from .utils import current_season

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

        return self.host.schedule(self.code, by='team')

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


    def boxscore(self, week=None, season=None):
        '''Boxscore stats for the game played in the specified week. If week=None
           return the current week from the nfl object
        '''

        if week:
            game = self.host.game(self.code, week, season)
        else:
            game = self.active_game

        if game is not None:
            return self.host.engine.boxscore(self.host, game)

    def plays(self, count=10, week=None, season=None):
        '''Returns most recent plays from the specified game.

           count:   number of most recent plays to return
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

        if game is not None:
            return self.host.engine.plays(self.host, game, count)

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
        champs = pd.concat([champs, pd.Series('', index=self.host.tiebreaks(self.teams-set(champs.index)).index)])

        # mark wildcard slots
        champs[4:7] = 'Wildcard'
        champs.name = self.code

        return champs[:count]


    def __repr__(self):
        return '{}\n'.format(self.code) + self.standings.__repr__()

    def _repr_html_(self):
        return '<h3>{}</h3>\n'.format(self.code) + self.standings._repr_html_()

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
    '''

    def __init__(self, year=None, season=None, engine=None):
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
        self.stash_ = None
        self.autoUpdate = True
        self.netTouchdowns = False

        if not self.year:
            self.year = current_season()

        if not engine:
            from .espn import NFLSourceESPN
            self.engine = NFLSourceESPN()
        # elif type(engine) is str:
        #     self.engine = getattr(sys.modules[__name__], engine)()

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

        key = meta.loc['Engine', 'value'].split('.')
        engine = getattr(sys.modules['.'.join(key[:-1])], key[-1])
        self.engine = engine()

        # resets and housecleaning
        self._post_load()
        return self

    def save(self, path):
        '''Save data to the specified Excel file
        '''

        teams = pd.DataFrame(columns=['code','name','conf','div']).set_index('code')
        meta  = pd.DataFrame(columns=['key', 'value']).set_index('key')

        for key,team in self.teams_.items():
            teams.loc[key, :] = [team['name'], team['conf'], team['div']]

        meta.loc['Last Updated', 'value'] = datetime.now()
        meta.loc['Engine', 'value'] = '.'.join([self.engine.__class__.__module__, self.engine.__class__.__name__])
        meta.loc['Year', 'value'] = self.year

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

    def stash(self, inplace=True):
        '''Saves a copy of current game data
        '''

        if len(self.games_) == 0:
            raise RuntimeError('game data has not yet been updated or loaded')

        stash_ = self.games_.copy()
        if inplace:
            self.stash_ = stash_
            return
        
        return stash_


    def restore(self, stash=None):
        ''' Restores from the previous stash
        '''

        if stash:
            if type(stash) is not pd.DataFrame:
                raise RuntimeError('object passed to restore() must be of type list')

            self.games_ = stash.copy()
            self.stats = None
            return

        if type(self.stash_) is not pd.DataFrame:
            raise RuntimeError('game data has not been previously stashed')

        self.games_ = self.stash_.copy()
        self.stats = None

    def update(self):
        ''' Updates team and game data from the underlying API
        '''

        self.teams_ = {}
        self.divs_  = {}
        self.confs_ = {}
        self.games_ = None

        for elem in self.engine.teams(self):
            key = elem['key']
            team = {
                'name': elem['name'],
                'conf': elem['conf'],
                'div':  '-'.join([elem['conf'], elem['div']])
            }
 
            self.teams_[key] = team

            (div,conf) = (team['div'],team['conf'])
            if self.divs_.get(div) is None:
                self.divs_[div] = set()

            if self.confs_.get(conf) is None:
                self.confs_[conf] = set()

            self.divs_[div].add(key)
            self.confs_[conf].add(key)

        # engine returns a flat dataframe
        g = self.engine.games(self)
        g['p'] = (g['hs'].isna()==False) & (g['as'].isna()==False)

        # convert nans to 0's so we can set column type to int
        g[['as','hs']] = g[['as','hs']].fillna(0).astype('int64')
        self.games_ = g.sort_values('ts').set_index(['seas', 'id'])
        self._post_load()
        return self

    def _post_load(self):
        '''Perform cleanup operations after an update or load
        '''

        self.seasons_ = {}
        teams = set(self.teams_.keys())
        for s in self.games_.index.get_level_values(0).unique():
            df = self.games_.xs(s)
            t = set(df['at'].unique()) | set(df['ht'].unique())
            self.seasons_[s] = t & teams

        if not self.week:
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

        self.stats = None

    @property
    def scoreboard(self):

        z = self.engine.scoreboard(self)
        if z and self.autoUpdate and self.year == z.year:
            for (k,v) in z.scoreboard.iterrows():
                if v['status'] == 'Final' or v['status'] == 'Final/OT':
                    self.set(z.week, **{v['hteam']: v['hscore'], v['ateam']: v['ascore']})

        return z

    @property
    def standings(self):

        return self._stats()[['name','div','overall','division','conference']]

    @property
    def NFC(self):

        return NFLConference('NFC', self)

    @property
    def AFC(self):

        return NFLConference('AFC', self)

    def set(self, wk, overwrite=True, ordered=False, season=None, **kwargs):
        ''' Set the final score(s) for games and weeks. You can use this to create
            hypothetical outcomes and analyze the effect on team rankings. Scores
            are specified by team code and applied to the specified week's schedule.
            If the score is specified for only one team in a game, the score of the other
            team is assumed to be zero if not previously set.

            wk:         week number or list-like of weeks
            overwrite:  overwrite previously set scores
            ordered:    scores specify outcomes only in order of priority
            **kwargs    dict of team codes and scores

            If for any given game only one team score is provided, the other is assumed to be zero.
            Use negative values to indicate 0-0 ties

            # typical examples

            set(7, MIN=17, GB=10)

            The above will set the score for the MIN/GB game in week 7 if
            MIN and GB play each other in that week. Otherwise, it will
            set each team's score respectively and default their opponent's scores
            to 0. Scores for bye teams in that week are ignored.

            set(range(16,19), PHI=1)

            Assuming week 15 is finished, set PHI to win its remaining games

            set(range(16, 19), PHI=0)
            set(range(16, 19), reset=False, WSH=0)

            Set PHI to lose its remaining games, and WSH to lose its remaining games, except
            against PHI. In that case, the first call would set WSH
            to win the head-to-head match, and the second call would ignore the WSH/PHI game
            because it had been previously set

            set(range(16, 19), ordered=True, DAL=0, WSH=1, PHI=0, NYG=1)

            Use the following hierarchy to set outcomes in weeks 16-18:
              1) DAL loses
              2) WSH wins
              3) PHI loses (except against DAL)
              4) NYG wins (except against WSH)
        '''

        # this mechanism allows keyword mapping to values
        value_map = {'win': 1, 'loss': 0, 'tie': -1}
        def _v(x):
            return value_map.get(x, x)

        # First argument can also be a series of scores or outcomes. Index must be a MultiIndex (week,team)
        if type(wk) is pd.Series:
            for w in wk.index.get_level_values(0).unique():
                self.set(w, reset, ordered, **wk.xs(w).to_dict())

        elif type(wk) is dict:
            # deprecated
            for k,v in wk.items():
                self.set(k, reset, ordered, **v)

            return

        if type(wk) is int:
            wk = [wk]

        teams = kwargs.keys()
        season = season or self.season
        inv = {0: 1, 1: 0, -1: 0}

        for (k,elem) in self.games_.iterrows():
            if k[0] != season or elem['wk'] not in wk:
                continue

            if elem['p'] and not overwrite:
                continue

            (ht,at) = (elem['ht'],elem['at'])
            if ordered:
                if ht in teams or at in teams:
                    for k,v in kwargs.items():
                        v = _v(v)
                        if k == ht:
                            self.games_.loc[k, ['hs', 'as', 'p']] = [max(v,0), 1 if v==0 else 0, True]
                            break
                        elif k == at:
                            self.games_.loc[k, ['as', 'hs', 'p']] = [max(v,0), 1 if v==0 else 0, True]
                            break

            elif ht in teams and at in teams:
                self.games_.loc[k, ['hs', 'as', 'p']] = [max(_v(kwargs[ht]),0), max(_v(kwargs[at]),0), True]

            elif ht in teams:
                v = _v(kwargs[ht])
                hscore = max(v,0)
                if v < 0:
                    ascore = hscore
                else:
                    ascore = 0 if hscore > 0 else 1

                self.games_.loc[k, ['hs', 'as', 'p']] = [hscore, ascore, True]

            elif at in teams:
                v = _v(kwargs[at])
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
        z = self.gameFrame(teams, week, season)[['p']]
        z.loc[:] = False
        self.games_.update(z.assign(season=season).reset_index().set_index(['season','id']))
        self.stats = None


    def gameFrame(self, teams=None, limit=None, allGames=False, season=None):
        '''Returns raw game data as a DataFrame. This is mostly used internally.
        '''

        season = season or self.season

        if type(limit) is int:
            limit = range(limit, limit+1)

        z = self.games_.xs(season)
        if limit:
            z = z[z['wk'].isin(limit)]

        if teams:
            teams = self._teams(teams)
            z = z[z['ht'].isin(teams) | z['at'].isin(teams)]

        if not allGames:
            z = z[z['p']]

        return z

    def games(self, teams=None, limit=None, allGames=False, season=None):
        '''Iterate over the game database, returning a series
        '''

        for (_,game) in self.gameFrame(teams, limit, allGames, season).iterrows():
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
                    Pass None for all weeks

        by:         Data orientation, either 'team' or 'game'
                        'game' will return unique games based on the team(s) you request.
                        If two of the teams you request play each other the result will
                        be one row not two

                        'team' will return teams as rows; if two requested teams play
                        each other there will be two rows not one, with inverse 'home'
                        and 'away' designations. Additionally, 'bye' teams will be listed as empty rows

        ts:         Return dates as Pandas datetime objects that can be used computationally,
                    otherwise (default) return an easily read string

        As a convenience, you can also pass an integer as the first argument to get the schedule
        for the specified week. The following two calls are the same:

        schedule(5)
        schedule(teams=None, weeks=5)

        Examples
          nfl.schedule('MIN', by='team')   # Vikings schedule
          nfl.schedule(weeks=range(3,8))   # schedule for weeks 3-6
          nfl.schedule(5)                  # shortcut for week 5: single ints are taken as a week number not a team

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
            teams2 = self._teams(teams)
            if type(teams) is str and len(teams2) > 1:
                # This is so we differentiate between a division/conference name and a team code
                teams = teams2

        weeks2 = self._teams(weeks)

        if by == 'game':
            games = self.gameFrame(teams=teams2, limit=weeks2, allGames=True, season=season).copy()
            df = games.rename({ 'wk': 'week', 'ht': 'hteam', 'at': 'ateam', 'hs': 'hscore', 'as': 'ascore', 'ts': 'date'}, axis=1)

            if (df['p']==False).any():
                # this assignment will change dtype to float64, so only do it if necessary
                df.loc[df['p']==False, ['ascore', 'hscore']] = np.nan

            df = df[['week', 'hteam', 'ateam', 'hscore', 'ascore', 'date']]
            if not ts:
                df['date'] = df['date'].dt.strftime('%m/%d %H:%M')

            if type(teams) in [str, NFLTeam] and type(weeks) is int:
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

        df = pd.DataFrame(index=pd.MultiIndex.from_product([weeks2, teams2], names=['week', 'team']),
                columns=['opp', 'loc', 'score', 'opp_score', 'wlt', 'date'])
        for game in self.games(teams=teams2, limit=weeks2, allGames=True, season=season):
            if game['ht'] in teams2:
                df.loc[(game['wk'],game['ht'])] = [game['at'], 'home', game['hs'], game['as'], NFL.result(game['hs'], game['as']), game['ts']]
            
            if game['at'] in teams2:
                df.loc[(game['wk'],game['at'])] = [game['ht'], 'away', game['as'], game['hs'], NFL.result(game['as'], game['hs']), game['ts']]

        df['date'] = df['date'].astype('datetime64[ns]')
        if not ts:
            df['date'] = df['date'].dt.strftime('%m/%d %H:%M')

        if type(teams) is str and type(weeks) is int:
            return df.loc[(weeks,teams)]
        elif type(teams) is str:
            return df.xs(teams, level=1)
        elif type(teams) is NFLTeam:
            return df.xs(teams.code, level=1)
        elif type(weeks) is int:
            return df.xs(weeks, level=0)

        return df


    def opponents(self, teams, limit=None, allGames=False, season=None):
        ''' Returns the set of common opponents of one or more teams

            The teams argument can be a single team or a list.
        '''

        if teams is None:
            raise ValueError("teams cannot be None here")

        teams = self._teams(teams)

        ops = {t:set() for t in teams}

        for game in self.games(teams, limit, allGames=allGames, season=season):
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

    def wlt(self, teams=None, within=None, limit=None, season=None):
        '''Return the wlt stats of one or more teams

        teams:  team code or list of team codes

        within: list of team codes that defines the wlt universe
        '''

        return self._wlt(teams=teams, within=within, limit=limit, season=season)[0].drop(['scored','allowed'], axis=1)

    def matrix(self, teams=None, limit=None, allGames=False, season=None):
        '''Return a matrix of teams and the number of games played against each other
        '''

        return self._wlt(teams, limit=limit, allGames=allGames, season=season)[1]


    def _wlt(self, teams=None, within=None, limit=None, allGames=False, season=None):
        ''' Internal function for calculating wlt from games database
        options to calculate ancillary data.

        points: include columns for points scored and allowed

        matrix: if True, returns the wlt frame and the games frame as a tuple
        '''

        teams  = self._teams(teams)
        within = self._teams(within)

        cols = ['win','loss','tie', 'pct', 'scored','allowed']

        df = pd.DataFrame(index=list(teams), columns=cols)
        df[df.columns] = 0
        df.columns.name = 'outcome'
        df.index.name = 'team'

        # define a matrix of games played against opponents
        m = pd.DataFrame(0, index=list(teams), columns=list(teams))
        for t in teams:
            m.loc[t, t] = np.nan

        for game in self.games(teams, limit, allGames=allGames, season=season):
            if game['ht'] in teams and (within is None or game['at'] in within):
                z = NFL.result(game['hs'], game['as'])
                if z:
                    df.loc[game['ht'], z] += 1
                    df.loc[game['ht'], 'scored'] += game['hs']
                    df.loc[game['ht'], 'allowed'] += game['as']

                if game['at'] in m.columns:
                    m.loc[game['ht'], game['at']] += 1

            if game['at'] in teams and (within is None or game['ht'] in within):
                z = NFL.result(game['as'], game['hs'])
                if z:
                    df.loc[game['at'], z] += 1
                    df.loc[game['at'], 'scored'] += game['as']
                    df.loc[game['at'], 'allowed'] += game['hs']


                if game['ht'] in m.columns:
                    m.loc[game['at'], game['ht']] += 1                


        df['pct'] = (df['win'] + df['tie'] * 0.5) / df.drop(columns=['scored','allowed'], errors='ignore').sum(axis=1)
        df.sort_values('pct', ascending=False, inplace=True)
        
        return (df, m)

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

        teams = self._teams(teams)
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([teams, ['win','loss','tie', 'pct']], names=['team','outcome']))
        common_opponents = self.opponents(teams)
        divisions = set()
        stats = self._stats()
        if self.netTouchdowns:
            ntd = self.engine.net_touchdowns(self, teams)
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

        (h2h,gm) = self._wlt(teams, within=teams, season='reg')
        co  = self._wlt(teams, within=common_opponents, season='reg')[0]

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
            if not gm.isin([np.nan, gm.iloc[0,1]]).all().all():
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


    def tiebreaks(self, teams, fast=False, divRule=True):
        '''Returns a series with the results of a robust tiebreaker analysis for teams
           Team codes comprise the series index in hierarchical order, while the series
           value indicates the rule (or basis) for each team besting the one below it

           teams     a list-like of team codes, or a division/conference code

           fast      If true, then only return the highest ranking team. The function
                     also takes measures to avoid expensive calculations if possible

           divRule   Enforce the "one-club-per-division" rule as stated in the wildcard
                     tiebreaking procedures
        '''
        teams = list(self._teams(teams))
        r = pd.Series(name='level')

        # shortcuts for efficiency: implement before call to _stats for speed
        if len(teams) == 0:
            return r
        elif len(teams) == 1:
            r[teams[0]] = 'overall'
            return r
        elif fast:
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

            suggest = '' if self.netTouchdowns else ' (try setting nfl.netTouchdowns=True)'
            raise RuntimeError("Can't resolve tiebreaker: teams are essentially equal.{}".format(suggest))

        while len(teams) > 1:
            if fast and len(r) > 0:
                return r

            logging.debug(msg('1.1 (Begin)', teams))

            # apply division tiebreakers if necessary. start by counting # of divisions
            divs = {}
            for elem in teams:
                d = self.teams_[elem]['div']
                if d not in divs:
                    divs[d] = set()

                divs[d].add(elem)
            
            subTeams = set(teams)
            if divRule and len(divs) > 1:
                #  multiple divisions: eliminate all but the top team from each
                for k,v in divs.items():
                    if len(v) > 1:
                        logging.debug(msg('2.1 (1 club/division)', v))
                        s = self.tiebreaks(v, fast=True)
                        subTeams -= v - {s.index[0]}

                if len(subTeams) < len(teams):
                    logging.debug(msg('2.2 (Pruned)', subTeams))

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
        '''Run a wildcard analysis on the specified teams
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
            df.loc[i] = self.tiebreaks(i.split()).index[:seeds]

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


    def _teams(self, teams):
        ''' Transforms scalar values into arrays. Iterable objects are,
            for the most part, returned unchanged. Scalar values are
            returned as single-element arrays. Recognized conference and
            division codes are returns as the corresponding list of team codes
        '''

        if teams is None:
            return None

        if type(teams) in [pd.Series, pd.Index]:
            # a list-like that needs to be cast
            return list(teams)

        if type(teams) is NFLTeam:
            return [teams.code]

        if type(teams) in [NFLDivision, NFLConference]:
            return list(teams.teams)

        if type(teams) is not str and hasattr(teams, '__iter__'):
            # assume no problems iterating
            return teams

        # below here assume a scalar
        if teams in self.divs_:
            # division code
            return self.divs_[teams]

        if teams in self.confs_:
            # conference code
            return self.confs_[teams]

        # single team code
        return [teams]

    def scenarios(self, weeks, teams, spots=1, ties=True):
        '''Returns a dataframe of scenarios and outcomes with the specified constraints

           This function is essentially a wrapper for NFLScenarioMaker with
           the most common use case, i.e. determining playoff spots. You
           can customize and optimize the model by implementing NFLScenarioMaker
           directly (see docs).


           weeks    weeks for which to generate scenarios

           teams    list-list of teams for which to generate scenarios

           spots    number of available playoff spots. If 0, then NFLConference.playoffs
                    is used to determine eligibility, instead of the
                    general-purpose tiebreaker procedure. A value
                    of 1 sets the analysis to "fast" mode, which is
                    substantially more efficient

           ties     whether to include ties as possible outcomes
        '''

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
        
        with NFLScenarioMaker(self, weeks, teams, ties) as gen:
            df = gen.frame(['playoffs'])
            for option in gen:
                x = len(df)
                df.loc[x] = option
                df.loc[x, 'playoffs'] = False
                self.set(option)
                if spots == 0:
                    p = conf.playoffs()
                    z = [('playoffs',i) for i in set(teams) & set(p.index)]
                    df.loc[x, z] = True
                else:
                    tb = self.tiebreaks(teams, fast=(spots==1))
                    z = [('playoffs',i) for i in tb.index[:spots]]
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


class NFLScenarioMaker():
    '''Facilitates generating and testing different win/lose/tie scenarios,
       typically to analyze the outcome on the playoff picture.

       If impelemented in a "with" context the object saves and restores
       the game state.

       Typical usage:

       weeks = [17, 18]
       teams = ['LAC', 'DEN', 'CIN', 'IND', 'MIA']
       with NFLScenarioMaker(nfl, weeks, teams, ties=False) as s:
          df = s.frame(['playoffs])
          for option in s:
             z = len(df)

    '''

    def __init__(self, nfl, weeks, teams, ties=True):
        self.nfl = nfl
        self.weeks = weeks
        self.teams = teams
        self.ties = ties
        self.stash = None
        self.completed = None

    def __enter__(self):
        self.weeks = self.nfl._teams(self.weeks)
        self.teams = self.nfl._teams(self.teams)
        self.stash = self.nfl.stash(inplace=True)
        self.completed = self.nfl.schedule(self.teams, self.weeks, by='team')['wlt'].replace('', np.nan).dropna().index
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.nfl.restore(self.stash)

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


        aresults = {'win': 'loss', 'loss': 'win', 'tie': 'tie'}

        nfl = self.nfl
        teams = nfl._teams(self.teams)
        weeks = nfl._teams(self.weeks)

        sch = nfl.schedule(teams, weeks, by='game')
        wlt = nfl.schedule(teams, weeks, by='team')['wlt'].replace('',np.nan)
        s = pd.Series(index=pd.MultiIndex.from_product([weeks, teams], names=['week','team']), dtype=str)
        values = ('win','loss','tie')
        if not self.ties:
            values = values[:-1]
        for row in outcomes([values] * len(sch)):
            sch['hscore'] = row
            for k,elem in sch.iterrows():
                if k in s.index:
                    s[k] = elem['hscore']
                
                if (k[0],elem['ateam']) in s.index:
                    s[(k[0],elem['ateam'])] = aresults[elem['hscore']]

            # skip scenarios that conflict with existing scores
            if (wlt.fillna(s) == s).all():
                yield s.drop(self.completed)


    def frame(self, extra=[]):
        '''Return a DataFrame structured to hold the set of scenarios. Typically
           you call this in advance of iterating over the object.

           extra specifiies additional level-0 column names

           Typical example:

           teams = {'DEN', 'MIA', 'LAC'}
           weeks = [17, 18]

           with nfl.scenarios(weeks, teams) as s:
               df = s.frame(['playoff'])

               # NB: computationally intense: up to 729 possibilities, could take 10m or more
               for option in s:
                  at = len(df)
                  df.loc[at] = option
                  nfl.loc[at, 'playoff'] = ''
                  p = nfl.AFC.playoffs()
                  z = [('playoff', i) for i in set(teams) & set(p.index)]
                  nfl.loc[at, z] = 'x'
        '''

        weeks = self.nfl._teams(self.weeks) + extra
        teams = self.nfl._teams(self.teams)
        df = pd.DataFrame(columns=pd.MultiIndex.from_product([weeks, teams], names=['week', 'team']))
        return df.drop(self.completed, axis=1)


class NFLScoreboard():
    '''A wrapper class for scoreboard display of live games. Typically you
       obtain or display this through the nfl.scoreboard propety.

       Class attributes:
            week        scoreboard week
            year        scoreboard year
            scoreboard  pandas DataFrame
    '''

    def __init__(self, nfl, year, week, scoreboard):
        self.nfl  = nfl
        self.year = year
        self.week = week
        self.scoreboard = scoreboard

    def __repr__(self):
        if type(self.scoreboard) is pd.DataFrame:
            return 'Week {}\n'.format(self.week) + self.scoreboard.drop('state',axis=1).__repr__()

        return ''

    def _repr_html_(self):
        if type(self.scoreboard) is pd.DataFrame:
            return '<h3>Week {}</h3>\n'.format(self.week) + self.scoreboard.drop('state',axis=1)._repr_html_()

        return ''

    def __call__(self, teams=None, limit=None):
        '''Return scoreboard for just the specified team(s).
        '''

        z = self.scoreboard
        if teams:
            teams = self.nfl._teams(teams)
            z = z[z['ateam'].isin(teams) | z['hteam'].isin(teams)]
        
        if limit:
            z = z[z['state']==limit]

        return z.drop('state',axis=1)

        
class NFLPlaysFrame(pd.DataFrame):

    def desc(self, pos=-1):
        '''Return the description of a single row. Negative values are treated
           as offsets (i.e. from the end of the frame) while non-negative values
           are treated as ordinary keys
        '''
        if pos < 0:
            return self.iloc[pos]['desc']

        return self.loc[pos]['desc']

class NFLRoster():
    '''Contains a team roster
    '''

    def __init__(self, roster):
        self.roster = roster

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
        '''If an int is passed, returns the player with that jersey number.
        If a str is passed, returns the portion of the roster for the given position code
        '''

        if type(key) is int:
            df = self.roster[self.roster['jersey']==str(key)]
            if len(df) > 0:
                return df.iloc[0].rename('player')

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
    def quarterback(self):
        '''Starting quarterback (i.e. 1st quarterback in roster)
        '''
        return self.member('QB')

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
        row = self[code].iloc[0]
        if row['jersey'] is np.nan:
            return row['name']
        
        return '{} ({})'.format(row['name'], row['jersey'])
