# nfl.py #

A "work-in-progress" module for working with NFL team stats in a pandas environment.

## Requirements ##

Developed using python 3.10+ along with the packages listed below

## Installation ##

This is a basic script (not a package) so you'll need to download
it yourself and install the package depedencies. The non-standard ones
are:

* docopt
* openpyxl
* requests
* pandas
* numpy

You may also need to modify the path to your python installation in the shebang path
(line 1 of nfl.py).

## Data Sources & Updates ##

The file `NFLData.xlsx` provides the default source of data:
schedule, scores and team information. `NFL.update` provides
a tool to scrape game scores for a given year from
<https://www.pro-football-reference.com>. At the moment there
is no facility to update team information so this
must be maintainaned manually. I'm not sure what happens
to the current schedule if the `update` tool is run mid-way
through the season, but I can fix that next year ;)

## Command Line Options ##

Run ./nfl.py -h for current options. This may expand in the future.

To update the database using the latest available scores through week 18, run this:

````
./nfl.py update NFLData.xlsx 10
````

At the moment this will clobber any remaining schedule if you enter an earlier week,
and it's unknown what will happen mid-season, so backups are a good idea
(again, to be fixed next year).

## Module Interface ##

The script can also be loaded as a module interface in the interpreter
or a script/notebook. Doc strings and help text is available for most methods.

````
from nfl import NFL
nfl = NFL().load()  # defaults to ./NFLData.xlsx
nfl('NFC')          # return NFC conference standings
````

You can use the syntax above to return (as objects) information about any
team, division or conference. You can also access a team's schedule like this:

````
nfl('BAL').schedule
````

## Tiebreakers ##

This module's primary purpose is to provide tools to analyze and better
understand the NFL's tiebreaker rules. For example, you can look at a division's
tiebreaker results like this:

````
>>> nfl.tiebreakers(nfl('NFC-South').teams)
````

Which (at the end of the 2024 season) returns this multi-index dataframe:

````
team                  TB                           CAR                             NO                           ATL                       
outcome              win   loss  tie        pct    win   loss  tie         pct    win   loss  tie        pct    win   loss  tie        pct
overall              9.0    8.0  0.0   0.529412    2.0   15.0  0.0    0.117647    9.0    8.0  0.0   0.529412    7.0   10.0  0.0   0.411765
head-to-head         4.0    2.0  0.0   0.666667    1.0    5.0  0.0    0.166667    4.0    2.0  0.0   0.666667    3.0    3.0  0.0   0.500000
division             4.0    2.0  0.0   0.666667    1.0    5.0  0.0    0.166667    4.0    2.0  0.0   0.666667    3.0    3.0  0.0   0.500000
common-games         5.0    3.0  0.0   0.625000    1.0    7.0  0.0    0.125000    3.0    5.0  0.0   0.375000    3.0    5.0  0.0   0.375000
conference           7.0    5.0  0.0   0.583333    1.0   11.0  0.0    0.083333    6.0    6.0  0.0   0.500000    4.0    8.0  0.0   0.333333
victory-strength    58.0   95.0  0.0   0.379085   17.0   17.0  0.0    0.500000   52.0  101.0  0.0   0.339869   55.0   64.0  0.0   0.462185
schedule-strength  139.0  150.0  0.0   0.480969  151.0  138.0  0.0    0.522491  125.0  164.0  0.0   0.432526  124.0  165.0  0.0   0.429066
conference-rank      NaN    NaN  NaN  11.000000    NaN    NaN  NaN    1.500000    NaN    NaN  NaN  14.000000    NaN    NaN  NaN   5.000000
overall-rank         NaN    NaN  NaN  21.500000    NaN    NaN  NaN    1.000000    NaN    NaN  NaN  27.500000    NaN    NaN  NaN   8.000000
common-netpoints     NaN    NaN  NaN  36.000000    NaN    NaN  NaN  -77.000000    NaN    NaN  NaN  -9.000000    NaN    NaN  NaN -36.000000
overall-netpoints    NaN    NaN  NaN  23.000000    NaN    NaN  NaN -180.000000    NaN    NaN  NaN  75.000000    NaN    NaN  NaN -52.000000
````

Each row represents an NFL tie-breaking rule. The rules provided and their order vary depending on
whether the specified teams are in the same division or not. While the win/loss/tie columns are often
interesting, it's really the "pct" columns that matter in determining tiebreakers. This column is also
used to store conference/overall rank and net points for the appropriate rules. Team ranks are shown
in ascending order (i.e. 1 is worst not best) for reasons that will be apparent further down.

Currently, "net touchdowns" is not scraped so that rule is not included (neither is "coin toss" obviously).

You can use pandas to produce a more consise version of the tiebreakers table, just looking at the 'pct' column:

````
>>> t = nfl.tiebreakers(nfl('NFC-South').teams)
>>> t.xs('pct', axis=1, level=1)
team                      TB         CAR         NO        ATL
overall             0.529412    0.117647   0.529412   0.411765
head-to-head        0.666667    0.166667   0.666667   0.500000
division            0.666667    0.166667   0.666667   0.500000
common-games        0.625000    0.125000   0.375000   0.375000
conference          0.583333    0.083333   0.500000   0.333333
victory-strength    0.379085    0.500000   0.339869   0.462185
schedule-strength   0.480969    0.522491   0.432526   0.429066
conference-rank    11.000000    1.500000  14.000000   5.000000
overall-rank       21.500000    1.000000  27.500000   8.000000
common-netpoints   36.000000  -77.000000  -9.000000 -36.000000
overall-netpoints  23.000000 -180.000000  75.000000 -52.000000
````

And you can sort the teams on tiebreaker rules like this (the ascending=False parameter puts the highest team on the left):

````
>>> t.xs('pct', axis=1, level=1).sort_values(list(t.index), axis=1, ascending=False)
team                      TB         NO        ATL         CAR
overall             0.529412   0.529412   0.411765    0.117647
head-to-head        0.666667   0.666667   0.500000    0.166667
division            0.666667   0.666667   0.500000    0.166667
common-games        0.625000   0.375000   0.375000    0.125000
conference          0.583333   0.500000   0.333333    0.083333
victory-strength    0.379085   0.339869   0.462185    0.500000
schedule-strength   0.480969   0.432526   0.429066    0.522491
conference-rank    11.000000  14.000000   5.000000    1.500000
overall-rank       21.500000  27.500000   8.000000    1.000000
common-netpoints   36.000000  -9.000000 -36.000000  -77.000000
overall-netpoints  23.000000  75.000000 -52.000000 -180.000000
````

Here, you can see that Tampa Bay wins the tiebreaker by virtue of having a better record than New Orleans in common games
(i.e. games where they played the same opponents).

The table above isn't quite correct because Atlanta and Carolina do not qualify, and therefore should
be excluded from head-to-head and "common" calculations. That's easily fixed as shown below, although the result
in this case turns out to be essentially the same:

````
>>> nfl.tiebreakers(['TB', 'NO'])
team                  TB                            NO                       
outcome              win   loss  tie        pct    win   loss  tie        pct
overall              9.0    8.0  0.0   0.529412    9.0    8.0  0.0   0.529412
head-to-head         1.0    1.0  0.0   0.500000    1.0    1.0  0.0   0.500000
division             4.0    2.0  0.0   0.666667    4.0    2.0  0.0   0.666667
common-games         8.0    4.0  0.0   0.666667    6.0    6.0  0.0   0.500000
conference           7.0    5.0  0.0   0.583333    6.0    6.0  0.0   0.500000
victory-strength    58.0   95.0  0.0   0.379085   52.0  101.0  0.0   0.339869
schedule-strength  139.0  150.0  0.0   0.480969  125.0  164.0  0.0   0.432526
conference-rank      NaN    NaN  NaN  11.000000    NaN    NaN  NaN  14.000000
overall-rank         NaN    NaN  NaN  21.500000    NaN    NaN  NaN  27.500000
common-netpoints     NaN    NaN  NaN  49.000000    NaN    NaN  NaN  38.000000
overall-netpoints    NaN    NaN  NaN  23.000000    NaN    NaN  NaN  75.000000
````

As [described by the NFL](https://www.nfl.com/standings/tie-breaking-procedures) the
tiebreaking procedure is iterative when more than two teams are involved, so a complete
analysis might involve a few restarts.

## "What-If" Analysis ##

A fun exercise in the final weeks of regular season play is to speculate on a team's playoff prospects
based on the outcome of games that haven't been played yet. For example, "Cleveland must win its last
two games and Pittsburgh must lose to Seattle to secure a wildcard spot..." blah blah.

You can load the latest data as described above, and use the `set()` tool to specify hypothetical
results for games that haven't been played yet. You can start by looking at the last 2 weeks of
the schedule:

````
>>> nfl.schedule(nfl('AFC-North').teams, [17, 18])
           opp at_home score opp_score wlt
week team                                 
17   BAL   MIA       1  None      None    
     CIN    KC       0  None      None    
     CLE   NYJ       1  None      None    
     PIT   SEA       0  None      None    
18   BAL   PIT       1  None      None    
     CIN   CLE       1  None      None    
     CLE   CIN       0  None      None    
     PIT   BAL       0  None      None
````

And then you can set hypothetical outcomes like this:

````
>>> nfl.set(17, CLE=3, PIT=0)      # shorthand for Cleveland wins their game by 3 and Pittsburgh loses in week 17
>>> nfl.set(17, BAL=20, MIA=14)    # or specify actual scores for both teams
>>> nfl.set(18, CLE=1)             # Cleveland wins (by 1 point)
>>> nfl.schedule(nfl('AFC-North').teams, [17, 18])
           opp at_home score opp_score   wlt
week team                                   
17   BAL   MIA       1    20        14   win
     CIN    KC       0  None      None      
     CLE   NYJ       1     3         0   win
     PIT   SEA       0     0         1  loss
18   BAL   PIT       1  None      None      
     CIN   CLE       1     0         1  loss
     CLE   CIN       0     1         0   win
     PIT   BAL       0  None      None  
````

Note that you don't have to know who the opposing teams are; the function will use implicit scores if missing (either
1 or 0) and so long as the tiebreakers don't go all the way down to the "net points" level the outcome will be the same.


Use `clear()` and `reload()` to clear scores for part of the schedule, or reload the database from its original
Excel file (wiping out any scores you specify):

````
>>> nfl.clear([17, 18]) # clear scores for the final 2 weeks
>>> nfl.reload()        # reload the database
````
