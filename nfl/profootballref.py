
from pyquery import PyQuery
import urllib
import logging
import pandas as pd
import numpy as np
import re

from .source import NFLSource
from datetime import datetime
from .utils import safeInt, to_seconds, to_int_list

class NFLSourceProFootballRef(NFLSource):

    source = 'https://www.pro-football-reference.com/years/{}/games.htm'

    def teams(self, nfl):
        '''Generator must return: 
                key (team code)
                name
                conf
                div
        '''

        url = 'https://www.pro-football-reference.com/years/{}/'.format(nfl.year)

        try:
            d = PyQuery(url=url)('#AFC tbody tr, #NFC tbody tr')
        except urllib.error.HTTPError as err:
            logging.error('Bad URL: {}'.format(url))
            raise

        conf = div = ''
        for row in d:
            if PyQuery(row).hasClass('thead'):
                (conf,div) = PyQuery(row).text().split(maxsplit=1)
            else:
                name = PyQuery(row)('th').text()
                code = _code_from_url(PyQuery(row)('th a').attr['href'])
                if code:
                    yield {
                        'key': code,
                        'name': re.sub(r'[^\w\s]', '', name),
                        'conf': conf,
                        'div': div
                    }

    def games(self, nfl):
        '''Must return a pandas DataFrame with game information. See espn.py
           for details
        '''

        url = self.source.format(nfl.year)
        df = pd.DataFrame(columns=['seas', 'id', 'wk', 'ts', 'at', 'ht', 'as','hs'])

        try:
            d = PyQuery(url=url)('#all_games #games tbody > tr')
        except urllib.error.HTTPError as err:
            logging.error('Bad URL: {}'.format(url))
            raise

        def rowval(elem, id, tag='td'):
            return PyQuery(elem)('{}[data-stat="{}"]'.format(tag, id)).text()

        self.lasturl = url
        for elem in d:
            week = safeInt(rowval(elem, "week_num", "th"))
            if type(week) is int:
                acode = _code_from_url(PyQuery(elem)('td[data-stat="winner"] a').attr('href'))
                hcode = _code_from_url(PyQuery(elem)('td[data-stat="loser"] a').attr('href'))
                ascore = safeInt(rowval(elem, "pts_win"))
                hscore = safeInt(rowval(elem, "pts_lose"))
                if type(ascore) is not int: ascore = np.nan
                if type(hscore) is not int: hscore = np.nan
                game_date = pd.to_datetime(' '.join([rowval(elem, "game_date"), rowval(elem, "gametime")]))

                if rowval(elem, "game_location") != '@':
                    (acode,hcode) = (hcode,acode)
                    (ascore,hscore) = (hscore,ascore)

                gameid = '{:4d}{:02d}{:02d}0{}'.format(game_date.year, game_date.month, game_date.day, hcode.lower())
                df.loc[len(df)] = ['reg', gameid, week, game_date, acode, hcode, ascore, hscore]

        return df


    def boxscore(self, nfl, game):
        '''Return a dataframe of games stats for the specified week. None is returned for bye weeks
        '''

        if game is not None:
            ht = game['ht']
            # alt = nfl.teams_[ht]['alt'] or ht
            # code = '{}0{}'.format(datetime.strftime(game['ts'], '%Y%m%d'), alt.lower())
            code = game.name

            setup = {
                'head': ['week', 'opp', 'date','id'],
                'points': ['q1', 'q2', 'q3', 'q4', 'ot', 'final'],
                'game': ['yards', '1st_downs', 'turnovers', 'time_of_poss', 'secs_of_poss'],
                'rushing': ['count', 'yds', 'tds'],
                'passing': ['comp', 'att', 'yds', 'yds_net', 'tds', 'int'],
                'fumbles': ['count', 'lost'],
                'sacks': ['count', 'yds_lost'],
                '3d_conv': ['count', 'of'],
                '4d_conv': ['count', 'of'],
                'penalties': ['count', 'yds'],
            }

            idx = pd.MultiIndex.from_arrays([
                list(np.concatenate([[k]*len(v) for k,v in setup.items()])),
                list(np.concatenate(list(setup.values())))
                ], names=['cat', 'key'])

            df = pd.DataFrame(index=idx, columns=[game['at'], game['ht']])
            df.loc[('head','week'), :] = game['wk']
            df.loc[('head','opp'), :] = [game['ht'], game['at']]
            df.loc[('head','date'), :] = datetime.strftime(game['ts'], '%Y-%m-%d')
            df.loc[('head','id'), :] = code

            if not game['p']:
                return df.dropna()
            
            self.lasturl = 'https://www.pro-football-reference.com/boxscores/{}.htm'.format(code)
            d = PyQuery(url=self.lasturl)


            t = PyQuery(d)('table.linescore')
            quarters = [PyQuery(elem).text().lower() for elem in PyQuery(t)('thead th')][2:]
            ateam    = [PyQuery(elem).text().lower() for elem in PyQuery(t)('tr:nth-child(1) td')][2:]
            hteam    = [PyQuery(elem).text().lower() for elem in PyQuery(t)('tr:nth-child(2) td')][2:]

            quarters[0:4] = ['q'+x for x in quarters[0:4]]
            for i in zip(quarters, ateam, hteam):
                df.loc[('points', i[0]), :] = [int(i[1]), int(i[2])]

            # #team_stats table is commented out, so we have to dig for it
            html = PyQuery(d)('#all_team_stats').html().replace('<!--', '').replace('-->', '')
            team_stats = PyQuery(html)('#team_stats')

            for tr in PyQuery(team_stats)('tr'):
                stat = PyQuery(tr)('th').eq(0).text().lower()
                at   = PyQuery(tr)('td').eq(0).text().lower()
                ht   = PyQuery(tr)('td').eq(1).text().lower()

                if stat == 'first downs':
                    df.loc[('game', '1st_downs'), :] = [int(at), int(ht)]
                elif stat == 'total yards':
                    df.loc[('game', 'yards'), :] = [int(at), int(ht)]
                elif stat == 'turnovers':
                    df.loc[('game', 'turnovers'), :] = [int(at), int(ht)]
                elif stat == 'penalties-yards':
                    at = to_int_list(at)
                    ht = to_int_list(ht)
                    df.loc[('penalties','count'), :] = [at[0], ht[0]]
                    df.loc[('penalties','yds'), :]   = [at[1], ht[1]]
                elif stat == 'sacked-yards':
                    at = to_int_list(at)
                    ht = to_int_list(ht)
                    df.loc[('sacks','count'), :] = [at[0], ht[0]]
                    df.loc[('sacks','yds_lost'), :]   = [at[1], ht[1]]
                elif stat == 'fumbles-lost':
                    at = to_int_list(at)
                    ht = to_int_list(ht)
                    df.loc[('fumbles','count'), :] = [at[0], ht[0]]
                    df.loc[('fumbles','lost'), :]   = [at[1], ht[1]]
                elif stat == 'rush-yds-tds':
                    at = to_int_list(at)
                    ht = to_int_list(ht)
                    df.loc[('rushing','count'), :] = [at[0], ht[0]]
                    df.loc[('rushing','yds'), :]   = [at[1], ht[1]]
                    df.loc[('rushing','tds'), :]   = [at[2], ht[2]]
                elif stat == 'cmp-att-yd-td-int':
                    at = to_int_list(at)
                    ht = to_int_list(ht)
                    df.loc[('passing','comp'), :] = [at[0], ht[0]]
                    df.loc[('passing','att'), :]   = [at[1], ht[1]]
                    df.loc[('passing','yds'), :]   = [at[2], ht[2]]
                    df.loc[('passing','tds'), :]   = [at[3], ht[3]]
                    df.loc[('passing','int'), :]   = [at[4], ht[4]]
                elif stat == 'net pass yards':
                    df.loc[('passing', 'yds_net'), :] = [int(at), int(ht)]
                elif stat == 'time of possession':
                    df.loc[('game', 'time_of_poss'), :] = [at, ht]
                    df.loc[('game', 'secs_of_poss'), :] = [to_seconds(at), to_seconds(ht)]
                elif stat in ['third down conv.', 'fourth down conv.']:
                    key = '3d_conv' if stat == 'third down conv.' else '4d_conv'
                    at = to_int_list(at)
                    ht = to_int_list(ht)
                    df.loc[(key,'count'), :] = [at[0], ht[0]]
                    df.loc[(key,'of'), :]   = [at[1], ht[1]]

            return df

def _code_from_url(url):
    m = re.match(r'/teams/(\w+)/', url)
    if m:
        return m.group(1).upper()