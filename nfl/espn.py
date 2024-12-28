
from datetime import datetime
from dateutil import tz
import requests
import numpy as np
import pandas as pd

from .source import NFLSource
from .nfl import NFLScoreboard, NFLRoster
from .utils import safeInt, to_seconds, to_int_list, current_season

class NFLSourceESPN(NFLSource):

    source = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={:04d}{:02d}'
    name   = 'ESPN'

    def __init__(self):
        self.zone = tz.gettz('America/New_York')

    def extra_fields(self, type):

        if type == 'games':
            return ['id']

    def teams(self, nfl):
        '''Generator must return: 
                key (team code)
                name
                conf
                div
        '''

        # ESPN provides two endpoints for teams, the more efficient one only seems
        # to return the current season, so we try that one first
        if nfl.year == current_season():
            src = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/groups'
            result = requests.get(src).json()['groups']
            for conf in result:
                for div in conf['children']:
                    (conf_name,div_name) = div['name'].split(maxsplit=1)
                    for team in div['teams']:
                        yield {
                            'key': team['abbreviation'],
                            'name': team['displayName'],
                            'conf': conf_name,
                            'div': div_name
                        }
        else:
            # for prior years it's much harder, because for some insane reason the team object doesn't include
            # the team's division, so we have to iterate over conferences, divisions and teams: much more expensive
            # NB: the URLs returned by the API are limited to 25 items by default, which is fine because no conference
            # or division has more children than that. If required you could hack the URL and change the page size like this:
            # https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/teams?limit=100

            src = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{}/types/2/groups/'.format(nfl.year)
            for conf_data in requests.get(src).json()['items']:
                conf = requests.get(conf_data['$ref']).json()
                for div_data in requests.get(conf['children']['$ref']).json()['items']:
                    div = requests.get(div_data['$ref']).json()
                    (conf_name,div_name) = div['name'].split(maxsplit=1)
                    for team_data in requests.get(div['teams']['$ref']).json()['items']:
                        team = requests.get(team_data['$ref']).json()
                        yield {
                            'key': team['abbreviation'],
                            'team': team['displayName'],
                            'conf': conf_name,
                            'div': div_name
                        }

    def games(self, nfl):
        '''Generator must return:
            week (int)
            date (gametime as datetime)
            ateam (code for away team)
            hteam (code for home team)
            ascore (int)
            hscore (int)
        '''

        season_type = 2
        (start,end) = self.season_dates(nfl.year, season_type)
        first = start.floor(freq='D').replace(day=1)
        last  = end.floor(freq='D').replace(day=1)

        d = first
        while d <= last:
            self.lasturl = self.source.format(d.year, d.month)
            result = requests.get(self.lasturl).json()
            for elem in result['events']:
                if elem['season']['type'] == season_type:

                    (hteam,ateam) = elem['competitions'][0]['competitors']
                    if hteam['homeAway'] != 'home':
                        (hteam,ateam) = (ateam,hteam)

                    if elem['competitions'][0]['status']['type']['completed']:
                        ascore = safeInt(ateam['score'])
                        hscore = safeInt(hteam['score'])
                    else:
                        ascore = hscore = None

                    yield {
                        'week': elem['week']['number'],
                        'date': self.to_datetime(elem['date']),
                        'ateam': ateam['team']['abbreviation'],
                        'hteam': hteam['team']['abbreviation'],
                        'ascore': ascore,
                        'hscore': hscore,
                        'id': elem['id']
                    }

            d += pd.DateOffset(months=1)

    def boxscore(self, nfl, code, week):

        def team_stats(t):

            z = t['team']
            z['stats'] = {}
            for elem in t['statistics']:
                z['stats'][elem['name']] = elem['displayValue']

            return z

        game = nfl.game(code, week)
        if game:
            url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}'
            self.lasturl = url.format(game['id'])
            result = requests.get(self.lasturl).json()

            setup = {
                'head': ['week', 'opp', 'date', 'event'],
                'points': ['q1', 'q2', 'q3', 'q4', 'ot', 'final'],
                'game': ['drives', 'yards', '1st_downs', 'turnovers', 'fumbles_lost', 'time_of_poss', 'secs_of_poss'],
                'rushing': ['count', 'yds', 'tds'],
                'passing': ['comp', 'att', 'yds_net', 'tds', 'int'],
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
            df.loc[('head','event'), :] = game['id']

            if not game['p']:
                return df.dropna()

            (hteam,ateam) = result['header']['competitions'][0]['competitors']
            if hteam['homeAway'] != 'home':
                (hteam,ateam) = (ateam,hteam)

            df.loc[('points','final'), :] = [safeInt(ateam['score']), safeInt(hteam['score'])]

            qtrs = list(df.index.get_loc_level('points')[1])
            for i in zip(qtrs, ateam['linescores'], hteam['linescores']):
                df.loc[('points', i[0]), :] = [safeInt(i[1]['displayValue']), safeInt(i[2]['displayValue'])]

            # unpack team statistics into something simpler
            (hstats,astats) = map(lambda x: x['statistics'], result['boxscore']['teams'])
            if result['boxscore']['teams'][0]['homeAway'] != 'home':
                (hstats,astats) = (astats,hstats)

            hstats = {elem['name']: elem['displayValue'] for elem in hstats}
            astats = {elem['name']: elem['displayValue'] for elem in astats}
            stats  = {k:[astats.get(k), hstats.get(k)] for k in hstats}

            df.loc[('game','drives'), :] = stats['totalDrives']
            df.loc[('game','yards'), :] = stats['totalYards']
            df.loc[('game','1st_downs'), :] = stats['firstDowns']
            df.loc[('game','turnovers'), :] = stats['turnovers']
            df.loc[('game', 'fumbles_lost'), :] = stats['fumblesLost']
            df.loc[('game','time_of_poss'), :] = stats['possessionTime']
            df.loc[('game','secs_of_poss'), :] = [to_seconds(stats['possessionTime'][0]), to_seconds(stats['possessionTime'][1])]

            at = to_int_list(stats['sacksYardsLost'][0])
            ht = to_int_list(stats['sacksYardsLost'][1])
            df.loc[('sacks','count'), :] = [at[0], ht[0]]
            df.loc[('sacks','yds_lost'), :]   = [at[1], ht[1]]

            at = to_int_list(stats['totalPenaltiesYards'][0])
            ht = to_int_list(stats['totalPenaltiesYards'][1])
            df.loc[('penalties','count'), :] = [at[0], ht[0]]
            df.loc[('penalties','yds'), :]   = [at[1], ht[1]]

            df.loc[('rushing','count'), :] = stats['rushingAttempts']
            df.loc[('rushing','yds'), :] = stats['rushingYards']
            df.loc[('rushing','tds'), :] = 0

            at = to_int_list(stats['completionAttempts'][0], '/')
            ht = to_int_list(stats['completionAttempts'][1], '/')

            df.loc[('passing','comp'), :] = [at[0], ht[0]]
            df.loc[('passing','att'), :] = [at[1], ht[1]]
            df.loc[('passing','yds_net'), :] = stats['netPassingYards']
            df.loc[('passing','int'), :] = stats['interceptions']
            df.loc[('passing','tds'), :] = 0

            at = to_int_list(stats['thirdDownEff'][0])
            ht = to_int_list(stats['thirdDownEff'][1])
            df.loc[('3d_conv', 'count'), :] = [at[0], ht[0]]
            df.loc[('3d_conv', 'of'), :] = [at[1], ht[1]]

            at = to_int_list(stats['fourthDownEff'][0])
            ht = to_int_list(stats['fourthDownEff'][1])
            df.loc[('4d_conv', 'count'), :] = [at[0], ht[0]]
            df.loc[('4d_conv', 'of'), :] = [at[1], ht[1]]

            # iterate scoringPlays to count touchdowns
            for score in result['scoringPlays']:
                key = game['ht'] if score['team']['uid'] == hteam['uid'] else game['at']
                if score['type']['id'] == '68':
                    df.loc[('rushing','tds'), key] += 1
                elif score['type']['id'] == '67':
                    df.loc[('passing','tds'), key] += 1

            return df

    def plays(self, nfl, code, week, count):

        game = nfl.game(code, week)
        if game:
            url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}'
            self.lasturl = url.format(game['id'])
            result = requests.get(self.lasturl).json()

            df = pd.DataFrame(columns=['team', 'period', 'clock', 'down', 'loc', 'yds', 'type', 'desc'])
            for drive in result['drives']['previous']:
                for play in drive['plays']:
                    df.loc[len(df)] = [drive['team']['abbreviation'], play['period']['number'], play['clock']['displayValue'],
                        play['start'].get('shortDownDistanceText',''),
                        play['start'].get('possessionText',''), play.get('statYardage',np.nan), play['type']['text'], play['text']
                    ]

            if count > 0:
                return df.iloc[-count:]

            return df


    def scoreboard(self, nfl):
        '''Return current scoreboard
        '''

        df = pd.DataFrame(columns=['ateam','hteam','ascore','hscore','period','clock','status','down','fpos','broadcast'])
        now = datetime.now()
        # url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={:04d}{:02d}{:02d}'
        # self.lasturl = url.format(now.year, now.month, now.day)
        # result = requests.get(self.lasturl).json()
        self.lasturl = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'

        def rget(obj, *argv):
            '''Safely retrieves values from a nested set of dicts
            '''
            if len(argv) == 0:
                return obj

            if type(obj) is dict:
                return rget(obj.get(argv[0]), *argv[1:])

            return None

        result = requests.get(self.lasturl).json()
        for elem in result['events']:
            game = elem['competitions'][0]
            (hteam,ateam) = game['competitors']
            if hteam['homeAway'] != 'home':
                (hteam,ateam) = (ateam,hteam)

            at = len(df)
            df.loc[at, ['hteam','ateam']] = [hteam['team']['abbreviation'], ateam['team']['abbreviation']]
            df.loc[at, ['hscore','ascore']] = [safeInt(hteam['score']), safeInt(ateam['score'])]
            df.loc[at, ['broadcast','down','fpos','period','clock']] = [game['broadcast'], '','','','']
            df.loc[at, 'status'] = game['status']['type']['detail'] # default
            status = game['status']['type']['state']
            if status == 'in':
                df.loc[at, ['period','clock']] = [game['status']['period'], game['status']['displayClock']]
                if type(game.get('situation')) is dict:
                    sit = game['situation']
                    if sit.get('possession'):
                        pos = sit['possession']
                        pos = df.loc[at, 'hteam'] if pos == hteam['id'] else df.loc[at, 'ateam']
                        df.loc[at, ['status','down','fpos']] = [pos, sit.get('shortDownDistanceText',''), sit.get('possessionText','')]
                    elif game['status']['type']['name'] == 'STATUS_HALFTIME':
                        df.loc[at, 'status'] = game['status']['type']['shortDetail']
                    elif rget(sit, 'lastPlay', 'drive', 'result'):
                        # between possessions: try to report result of last play
                        pos = sit['lastPlay']['team']['id']
                        pos = df.loc[at, 'hteam'] if pos == hteam['id'] else df.loc[at, 'ateam']
                        df.loc[at, 'status'] = '{} {}'.format(sit['lastPlay']['drive']['result'], pos)
                    else:
                        df.loc[at, 'status'] = rget(sit, 'lastPlay', 'type', 'text') or 'na'

            elif status == 'pre':
                df.loc[at, ['hscore','ascore']] = np.nan
                gametime = pd.to_datetime(game['date']).astimezone(self.zone).replace(tzinfo=None)
                df.loc[at, 'status'] = '{}/{} {:02d}:{:02d}'.format(gametime.month, gametime.day, gametime.hour, gametime.minute)

        return NFLScoreboard(result['season']['year'], result['week']['number'], df)


    def roster(self, nfl, code):
        '''Return team roster
        '''

        url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{}/roster'
        df = pd.DataFrame(columns=['side', 'pos', 'position', 'name', 'jersey'])
        sides = {'offense': '1:OFF', 'defense': '2:DEF', 'specialTeam': '3:SPEC', 'injuredReserveOrOut': '4:INJURED'}
        results = requests.get(url.format(code)).json()

        df.loc[len(df)] = ['0:COACH', 'COACH', 'Coach', '{} {}'.format(results['coach'][0]['firstName'], results['coach'][0]['lastName']), np.nan]
        for side in results['athletes']:
            if sides.get(side['position']):
                for elem in side['items']:
                    df.loc[len(df)] = [sides[side['position']], elem['position']['abbreviation'], elem['position']['name'], elem['fullName'], elem['jersey'] or np.nan]

        df.sort_values(['side', 'position'], inplace=True)
        return NFLRoster(df.replace({'side': r'.+:(.+)'}, {'side': r'\1'}, regex=True))

    def to_datetime(self, date):
        '''Returns string converted to a zoneless datetime in the current time zone
        '''
        return pd.to_datetime(date).to_pydatetime().astimezone(self.zone).replace(tzinfo=None)

    def season_dates(self, year, type=2):
        '''Returns span for a season (regular season = 2) as pandas datetimes
        '''

        url = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{}/types/{}'.format(year, type)
        result = requests.get(url).json()
        return (pd.to_datetime(result['startDate']), pd.to_datetime(result['endDate']))