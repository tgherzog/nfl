#!/usr/local/bin/python -u

"""
NFL stats: a program to calculate NFL team standings

Script is now designed as a module. To get started, use the command line options or load
interactively and try:

from nfl import NFL
nfl = NFL()
nfl.load('NFLData.xlsx')

nfl('MIN')                       # info and standings for vikings
nfl('NFC')                       # or the NFC conference
nfl.wlt(['MIN', 'GB'])           # win/loss/tie info
nfl.tiebreakers(['DAL', 'PHI'])  # tiebreaker calculations

# and lots more, and yet lots of room for enhancements/improvements ;)

Usage:
    nfl.py update FILE WEEK
    nfl.py team TEAM [--file FILE]
    nfl.py tiebreakers TEAMS... [--file FILE]

Commands:
    update          Update the database FILE, scraping data up to and including WEEK

    team            Report TEAM stats and schedule (division or conference work too)

    tiebreakers     Report tiebreaker analysis for the specified TEAMS

Options:
    --file FILE     Use FILE as database path [default: NFLData.xlsx]

"""


import openpyxl
import pprint
import requests
import sys
import re
import logging
import time
from docopt import docopt
from pyquery import PyQuery
import urllib
from datetime import datetime
import numpy as np
import pandas as pd

class NFLTeam():
    def __init__(self, code, host):

        elem = host.teams_[code]
        self.code = code
        self.name = elem['name']
        self.div  = elem['div']
        self.conf = elem['conf']
        self.host = host

    @property
    def division(self):
        '''Return array of teams in this team's division
        '''
        return self.host.divs_[self.div]

    @property
    def conference(self):
        '''Return array of teams in this team's conference
        '''
        return self.host.confs_[self.conf]

    @property
    def standings(self):
        '''Return team standings
        '''

        return self.host.team_stats(self.code)

    @property
    def schedule(self):
        '''Return the team's schedule
        '''

        return self.host.schedule(self.code)

    @property
    def opponents(self):
        '''The team's opponents
        '''

        return self.host.opponents(self.code)

    def __repr__(self):
        return '{}: {} ({})\n'.format(self.code, self.name, self.div) + self.standings.__repr__()

    def _repr_html_(self):
        return '<h3>{}: {} ({})</h3>\n'.format(self.code, self.name, self.div) + self.standings._repr_html_()


class NFLDivision():
    def __init__(self, code, host):

        self.code = code
        self.host = host
        self.teams = host.divs_[code]

    @property
    def standings(self):
        ''' Return division standings

            rank: if True, add 'rank' column based on in-division tiebreaker rules (longer)
                  if False, sort results by game wlt
        '''

        z = self.host._stats()
        z = z[z['div']==self.code][['overall','division']]
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

        z = self.host._stats()
        z = z[z['conf']==self.code][['div', 'overall','division', 'conference']]
        return z


    def playoffs(self, count=7):
        '''Return the current conference seeds in order
        '''

        # get top-listed teams from each division
        champs = self.standings.reset_index().groupby(('div','')).first()['team']

        # restructure and reorder by seed
        champs = pd.Series(champs.index, index=champs)
        champs = champs.reindex(self.host.tiebreaks(champs.index).index)

        # pull wildcards from the remaining teams
        wc = self.host.tiebreaks(self.teams - set(champs.index))
        for elem in wc.index[:count-len(champs)]:
            champs[elem] = 'Wildcard'

        return champs



    def __repr__(self):
        return '{}\n'.format(self.code) + self.standings.__repr__()

    def _repr_html_(self):
        return '<h3>{}</h3>\n'.format(self.code) + self.standings._repr_html_()

class NFL():

    def __init__(self):
        self.teams_  = {}
        self.divs_   = {}
        self.confs_  = {}
        self.games_  = []
        self.path   = None
        self.max_week = 0
        self.stats = pd.DataFrame()

    def __call__(self, i):

        if i in self.confs_:
            return NFLConference(i, self)

        if i in self.divs_:
            return NFLDivision(i, self)

        if i in self.teams_:
            return NFLTeam(i, self)

    def load(self, path='NFLData.xlsx'):
        ''' Loads data from the specified excel file

            path:   path to Excel file
        '''

        # in case of reload
        self.path   = path
        self.games_ = []
        self.teams_ = {}
        self.divs_  = {}
        self.confs_ = {}
        self.max_week = 0                  

        wb = openpyxl.load_workbook(path, read_only=True)
        for row in wb['Divisions']:
            team = row[0].value
            conf = row[2].value
            div = row[3].value
            if team and conf and div:
                div = '-'.join([conf, div])
                self.teams_[team] = {'name': row[1].value, 'div': div, 'conf': conf}
                if self.divs_.get(div) is None:
                    self.divs_[div] = set()

                self.divs_[div].add(team)

                if self.confs_.get(conf) is None:
                    self.confs_[conf] = set()

                self.confs_[conf].add(team)


        for row in wb['Scores']:
            if row[0].row > 1 and row[0].value:
                # at/ht = away team/home team - same for scores
                game = {'wk': row[0].value, 'at': row[1].value, 'as': row[2].value, 'ht': row[3].value, 'hs': row[4].value}
                game['p'] = game['as'] is not None and game['hs'] is not None
                self.games_.append(game)
                self.max_week = max(self.max_week, game['wk'])

        self.stats = None
        return self

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

        sched = {k:[] for k in self.teams_.keys()}

        def tally(hcode, acode, hscore, ascore, cat):

            if hscore > ascore:
                stats.loc[hcode, (cat,'win')] += 1
                stats.loc[acode, (cat,'loss')] += 1
                if cat == 'overall':
                    sched[hcode] += [(acode,1)]
                    sched[acode] += [(hcode,0)]
            elif hscore < ascore:
                stats.loc[hcode, (cat,'loss')] += 1
                stats.loc[acode, (cat,'win')] += 1
                if cat == 'overall':
                    sched[hcode] += [(acode,0)]
                    sched[acode] += [(hcode,1)]
            elif hscore == ascore:
                stats.loc[hcode, (cat,'tie')] += 1
                stats.loc[acode, (cat,'tie')] += 1
                if cat == 'overall':
                    sched[hcode] += [(acode,0)]
                    sched[acode] += [(hcode,0)]                   

        for k,team in self.teams_.items():
            stats.loc[k, ['name', 'div', 'conf']] = (team['name'], team['div'], team['conf'])
            stats.loc[k, ['overall','division','conference', 'vic_stren', 'sch_stren', 'misc']] = 0

        for game in self.games_:
            if game['p']:
                tally(game['ht'], game['at'], game['hs'], game['as'], 'overall')
                stats.loc[game['ht'], ('misc', 'pts-scored')] += game['hs']
                stats.loc[game['at'], ('misc', 'pts-scored')] += game['as']
                stats.loc[game['ht'], ('misc', 'pts-allowed')] += game['as']
                stats.loc[game['at'], ('misc', 'pts-allowed')] += game['hs']

                if self.teams_[game['ht']]['div'] == self.teams_[game['at']]['div']:
                    tally(game['ht'], game['at'], game['hs'], game['as'], 'division')

                if self.teams_[game['ht']]['conf'] == self.teams_[game['at']]['conf']:
                    tally(game['ht'], game['at'], game['hs'], game['as'], 'conference')
                    stats.loc[game['ht'], ('misc', 'conf-pts-scored')] += game['hs']
                    stats.loc[game['at'], ('misc', 'conf-pts-scored')] += game['as']
                    stats.loc[game['ht'], ('misc', 'conf-pts-allowed')] += game['as']
                    stats.loc[game['at'], ('misc', 'conf-pts-allowed')] += game['hs']

        # strength of victory/schedule
        for (k,row) in stats.iterrows():
            for (op,win) in sched[k]:
                stats.loc[k, 'sch_stren'] = (stats.loc[k, 'sch_stren'] + stats.loc[op, 'overall']).values
                if win == 1:
                    stats.loc[k, 'vic_stren'] = (stats.loc[k, 'vic_stren'] + stats.loc[op, 'overall']).values

        # temporary table for calculating ranks - easier syntax
        t = pd.concat([stats['conf'], stats['misc'][['pts-scored','pts-allowed']]], axis=1)
        stats[('misc','rank-overall')] = (t['pts-scored'].rank() + t['pts-allowed'].rank(ascending=False)).rank()

        t['conf-off-rank'] = t.groupby('conf')['pts-scored'].rank()
        t['conf-def-rank'] = t.groupby('conf')['pts-allowed'].rank(ascending=False)
        t['conf-rank'] = t['conf-off-rank'] + t['conf-def-rank']
        stats[('misc', 'rank-conf')] = t.groupby('conf')['conf-rank'].rank()

        for i in ['overall','division','conference', 'sch_stren', 'vic_stren']:
            stats[(i,'pct')] = (stats[(i,'win')] + stats[(i,'tie')]*0.5) / stats[i].sum(axis=1)

        # sort by division tiebreakers
        self.stats = stats           # so that tiebreaks doesn't go recursive

        s = pd.Series(index=stats.index)
        for div in stats['div'].unique():
            z = self.tiebreaks(div)
            z[:] = range(len(z))
            s.loc[z.index] = z.astype(s.dtype)

        self.stats = self.stats.assign(divrank=s).sort_values(['div','divrank']).drop('divrank', level=0, axis=1)
        return self.stats


    def reload(self):
        ''' Reloads the previous Excel file
        '''

        if self.path:
            self.load(self.path)

        return self

    def update(self, path, week, year=None):
        ''' Updates the Scores sheet of the specified Excel file by scraping the URL shown in code.
            The file must already exist and have a valid "Divisions" sheet with team names consistent
            with those on the source website.

            path:   path to Excel workbook

            week:   week number (1-based) of last week to load

            year:   season to load; else infer the latest season from the current date
        '''

        if not year:
            dt = datetime.now()
            year = dt.year if dt.month >= 8 else dt.year-1

        self.load(path)
        tids = {v['name']:k for k,v in self.teams_.items()}
        wb = openpyxl.load_workbook(path, read_only=False)
        ws = wb['Scores']
        row = 2

        def safeInt(i):
            try:
                i = int(i)
            except ValueError:
                pass

            return i

        for w in range(1, week+1):
            url = 'https://www.pro-football-reference.com/years/{}/week_{}.htm'.format(year, w)
            try:
                time.sleep(1) # avoid throttling
                d = PyQuery(url=url)('div.game_summaries div.game_summary table.teams')
            except urllib.error.HTTPError as err:
                logging.error('Bad URL: {}'.format(url))
                raise

            logging.info('Processing {}/{}: ({}) - {} games'.format(w, year, url, len(d)))
            for elem in d:
                ateam = PyQuery(PyQuery(elem)('tr:nth-child(2)'))
                bteam = PyQuery(PyQuery(elem)('tr:nth-child(3)'))
                (aname,ascore) = (ateam('td:nth-child(1)').text(), ateam('td:nth-child(2)').text())
                (bname,bscore) = (bteam('td:nth-child(1)').text(), bteam('td:nth-child(2)').text())

                try:
                    aname = tids[aname]
                    bname = tids[bname]
                except KeyError as k:
                    logging.error('{}: unrecognized team name: check the Divisions table'.format(k))
                    raise

                ws.cell(row=row, column=1, value=w)
                ws.cell(row=row, column=2, value=aname)
                ws.cell(row=row, column=3, value=safeInt(ascore))
                ws.cell(row=row, column=4, value=bname)
                ws.cell(row=row, column=5, value=safeInt(bscore))
                row += 1

        # empty the remaining cells
        for row in range(row,ws.max_row):
            for c in range(1,6):
                ws.cell(row=row, column=c, value=None)

        wb.save(path)
        return self

    def set(self, wk, **kwargs):
        ''' Set the final score(s) for games in a given week. You can use this to create
            hypothetical outcomes and analyze the effect on team rankings. Scores
            are specified by team code and applied to the specified week's schedule.
            If the score is specified for only one team in a game, the score of the other
            team is assumed to be zero if not previously set.

            wk:         week number
            **kwargs    dict of team codes and final scores

            # typical example
            set(7, MIN=17, GB=10)

            The above will set the score for the MIN/GB game in week 7 if
            MIN and GB play each other in that week. Otherwise, it will
            set each team's score respectively and default their opponent's scores
            to 0. Scores for bye teams in that week are ignored.
        '''

        # sanity checks
        bogus = set(kwargs.keys()) - set(self.teams_.keys())
        if len(bogus) > 0:
            raise KeyError('Invalid team codes: {}'.format(','.join(bogus)))       

        for elem in self.games_:
            if wk == elem['wk']:
                if elem['ht'] in kwargs:
                    elem['hs'] = kwargs[elem['ht']]
                    elem['as'] = elem['as'] or (0 if elem['hs'] > 0 else 1)

                if elem['at'] in kwargs:
                    elem['as'] = kwargs[elem['at']]
                    elem['hs'] = elem['hs'] or (0 if elem['as'] > 0 else 1)

            elif wk < elem['wk']:
                # assuming elements are sorted by week, we can stop at this point
                break

        self.stats = None # signal to rebuild stats

    def clear(self, week, teams=None):
        '''Clear scores for a given week or weeks

        week:   can be an integer, range or list-like. Pass None to clear all (for whatever reason)
        '''

        if type(week) is int:
            week = [week]

        for elem in self.games_:
            if week is None or elem['wk'] in week:
                elem['p'] = False
                elem['hs'] = elem['as'] = None

        self.stats = None


    def games(self, teams=None, limit=None, allGames=False):
        ''' generator to iterate over score data

            teams:      code or list-like of teams to fetch

            limit:      range or list-like of weeks to fetch. Integers are converted
                        to the top limit of a range

            Example:

                for score in scores('MIN', limit=10) # fetch Vikings record up to but not including week 10
        '''

        if type(limit) is int:
            limit = range(1, limit)

        teams = self._teams(teams)

        for elem in self.games_:
            if limit is None or elem['wk'] in limit:
                if teams is None or elem['at'] in teams or elem['ht'] in teams:
                    if elem['p'] or allGames:
                        yield elem


    def scores(self, teams=None, limit=None):
        ''' Returns interated game data structured by teams

            Result is  dict keyed by team code each of game results as follows:

            [wlt, us, them, op, home, week]

            wlt:  'win' 'loss' or 'tie'
            us:   our final score
            them: their final score
            op:   opponent (team code)
            home: True for home games
            week: week number
        '''

        if teams is None:
            z = {i:[] for i in self.teams_.keys()}
        else:
            teams = self._teams(teams)
            z = {i:[] for i in teams}

        for game in self.games(teams, limit):
            if teams is None or game['at'] in teams:
                z[game['at']].append([NFL.result(game['as'], game['hs']), game['as'], game['hs'], game['ht'], False, game['wk']])

            if teams is None or game['ht'] in teams:
                z[game['ht']].append([NFL.result(game['hs'], game['as']), game['hs'], game['as'], game['at'], True, game['wk']])

        return z


    def schedule(self, which, limit=None):
        ''' Return schedule for a week or team. Specifying multiple weeks/teams
            returns a multi-indexed data frame

            which:    a week number, team code or list-like of same.

            limit:    a team code, week number or list-like of same (opposite of which)
        '''

        limit = self._teams(limit) # turns out this syntax sugarizer works for lists of week numbers too

        if type(which) is int:
            df = pd.DataFrame(columns=['at', 'hscore', 'ascore'])
            df.index.name = 'ht'
            for game in self.games(teams=limit, limit=range(which,which+1), allGames=True):
                df.loc[game['ht']] = [game['at'], game['hs'], game['as']]
        elif type(which) is str:
            # pre-populate the index so the schedule includes the bye week
            if limit is None:
                limit = range(1, self.max_week+1)

            df = pd.DataFrame(columns=['opp', 'at_home', 'score', 'opp_score', 'wlt'], index=limit)
            df.index.name = 'week'
            for game in self.games(teams=which, limit=limit, allGames=True):
                if game['ht'] == which:
                    df.loc[game['wk']] = [game['at'], 1, game['hs'], game['as'], NFL.result(game['hs'],game['as'])]
                else:
                    df.loc[game['wk']] = [game['ht'], 0, game['as'], game['hs'], NFL.result(game['as'], game['hs'])]
        else:
            # support multiple teams with recursive calls
            z = {}
            for elem in which:
                z[elem] = self.schedule(elem, limit=limit)

            df = pd.concat(z)
            if df.index.names[1] == 'week':
                # ensure result is primarily ordered by week
                df = df.reorder_levels([1, 0]).sort_index()
                df.index.names = ['week', 'team']
            else:
                df.index.names = ['week', 'ht']

        return df

    def opponents(self, teams, limit=None):
        ''' Returns the set of common opponents of one or more teams

            The teams argument can be a single team or a list.
        '''

        if teams is None:
            raise ValueError("teams cannot be None here")

        teams = self._teams(teams)

        ops = {t:set() for t in teams}

        for game in self.games(teams, limit):
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

    def wlt(self, teams=None, within=None, limit=None):
        '''Return the wlt stats of one or more teams

        teams:  team code or list of team codes

        within: list of team codes that defines the wlt universe
        '''

        return self._wlt(teams=teams, within=within, limit=limit)[0].drop(['scored','allowed'], axis=1)

    def matrix(self, teams=None, limit=None, allGames=False):
        '''Return a matrix of teams and the number of games played against each other
        '''

        return self._wlt(teams, limit=limit, allGames=allGames)[1]


    def _wlt(self, teams=None, within=None, limit=None, allGames=False):
        ''' Internal function for calculating wlt from games database with
        options to calculate ancillary data.

        points: include columns for points scored and allowed

        matrix: if True, returns the wlt frame and the games frame as a tuple
        '''

        teams = self._teams(teams)

        cols = ['win','loss','tie', 'pct', 'scored','allowed']

        df = pd.DataFrame(index=list(teams), columns=cols)
        df[df.columns] = 0
        df.columns.name = 'outcome'
        df.index.name = 'team'

        # define a matrix of games played against opponents
        m = pd.DataFrame(index=list(teams), columns=list(teams))
        m[m.columns] = 0
        for t in teams:
            m.loc[t, t] = np.nan

        for game in self.games(teams, limit, allGames):
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

        (h2h,gm) = self._wlt(teams, within=teams)
        co  = self._wlt(teams, within=common_opponents)[0]

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


    def tiebreaks(self, teams):
        '''Returns a series with the results of a robust tiebreaker analysis for teams
           The series index will be in reverse elimination order, i.e. the winner (last
           eliminated team) ordered first and the loser (first eliminated team) ordered last.
           The series value corresponds to the reason each team was eliminated, relative to
           the team just before it.
        '''
        teams = list(self._teams(teams))
        r = pd.Series(name='eliminated-by')
        stats = self._stats()
    
        if len(teams) == 0:
            return r
        
        def subdiv(teams):
            '''Returns a dict for the corresponding divisions of the specified teams. Dict
               keys are division codes and values are arrays of correponding team codes
            '''

            s = {}
            for elem in teams:
                d = self.teams_[elem]['div']
                if d not in s:
                    s[d] = []

                s[d].append(elem)

            return s

        def check_until_same(s):
            '''Returns the number of initial elements with the same value as the next one
            '''

            if s.isna().all() or (s == np.inf).any():
                return len(s) - 1
                
            i = 0
            for (k,v) in s.diff(-1).items():
                if v != 0:
                    return i

                i += 1

            return len(s)-1

        while len(teams) > 1:
            z = stats.loc[teams,('overall','pct')].sort_values().copy()
            t = check_until_same(z)
            if t == 0:
                r[z.index[0]] = 'overall'
                teams.remove(z.index[0])
            else:
                # check out many divisions we're about to compare
                subTeams = list(z.index[0:t+1])
                divisions = subdiv(subTeams)
    
                # may need to eliminate some teams if there are multiple divisions
                # (first team from each division only)
                if len(divisions) > 1:
                    for k,v in divisions.items():
                        if len(v) > 1:
                            s = self.tiebreaks(v)
                            
                            for t2 in reversed(s.index[1:]):
                                r[t2] = 'division-tiebreaker:' + s[t2]
                                teams.remove(t2)
                                subTeams.remove(t2)                       
                    
                z = self.tiebreakers(subTeams).xs('pct', level=1, axis=1).T
                z = z.sort_values(list(z.columns)).drop(z.columns[0], axis=1)
                z_len = len(teams)
                for rule in z.columns:
                    t = check_until_same(z[rule])
                    if t == 0:
                        r[z.index[0]] = rule
                        teams.remove(z.index[0])
                        break
    
                if z_len == len(teams):
                    raise NotImplementedError("Can't resolve tiebreaker {}: teams are essentially equal".format(len(r)))
    
        # should always be one team left
        r[teams[0]] = 'winner'
    
        # return series in reverse index order (best team first)
        return r.reindex(r.index[::-1])

    @staticmethod
    def result(a, b):
        
        if a is None or b is None:
            return ''
        elif a > b:
            return 'win'
        elif a < b:
            return 'loss'

        return 'tie'

    def _teams(self, teams):
        ''' Transforms teams into an array of actual team codes, or None
        '''

        if teams is None:
            return None

        if type(teams) in [list, set, range]:
            # already a list-like
            return teams

        if type(teams) in [pd.Series, pd.Index]:
            # a list-like that needs to be cast
            return list(teams)

        # below here assume a scalar

        if teams in self.divs_:
            # division code
            return self.divs_[teams]

        if teams in self.confs_:
            # conference code
            return self.confs_[teams]

        # single team code
        return [teams]


if __name__ == '__main__':
    config = docopt(__doc__)
    # print(config)
    # sys.exit(0)

    if config['update']:
        logging.basicConfig(level=logging.INFO)

        NFL().update(config['FILE'], week=int(config['WEEK']))

    if config['team']:
        nfl = NFL()
        nfl.load(config['--file'])
        team = nfl(config['TEAM'])
        print(team)
        if type(team) is NFLTeam:
            print(team.schedule)

    if config['tiebreakers']:
        nfl = NFL()
        print(nfl.load(config['--file']).tiebreakers(config['TEAMS']))


