# Task — swe_agent_pilot_s_04 (joke2k__django-environ-174)

Source: extracted from the leading environment turn of the upstream trajectory.

We're currently solving the following issue within our repository. Here's the issue text:
ISSUE:
Database url.scheme only works for built-in schemes (backends)
I haven't find any issue related to this topic, so sorry in advance if it is something that has been already discussed.

According to this code inside ```Env.db_url_config()```:
``` python
class Env(object):
    # [...]
   @classmethod
    def db_url_config(cls, url, engine=None):
        # [...]
        if engine:
            config['ENGINE'] = engine
        if url.scheme in Env.DB_SCHEMES:
            config['ENGINE'] = Env.DB_SCHEMES[url.scheme]


        if not config.get('ENGINE', False):
            warnings.warn("Engine not recognized from url: {0}".format(config))
            return {}
```

Actually, the only way to set up a custom backend for a database is using the ```engine``` parameter when we call ```Env.db()``` at the same time that we put an invalid ```url.scheme``` in ```DATABASE_URL``` (what is weird) or we put our custom one again (that it is going to be considered invalid anyway because it doesn't exist in ```Env.DB_SCHEMES```), what is redundant.

``` python
import environ
env = environ.Env()
env.db('DATABASE_URL', engine='path.to.my.custom.backend')
# Having DATABASE_URL=path.to.my.custom.backend://user:password@host:post/database
```

This code fix this situation, simply letting pass the custom scheme as a valid one if it doesn't match one of the built-in ```DB_SCHEMES```.

``` python
if url.scheme in Env.DB_SCHEMES:
    config['ENGINE'] = Env.DB_SCHEMES[url.scheme]
else:
    config['ENGINE'] = url.scheme
```


INSTRUCTIONS:
Now, you're going to solve this issue on your own. Your terminal session has started and you're in the repository's root directory. You can use any bash commands or the special interface to help you. Edit all the files you need to and run any checks or tests that you want.
Remember, YOU CAN ONLY ENTER ONE COMMAND AT A TIME. You should always wait for feedback after every command.
When you're satisfied with all of the changes you've made, you can submit your changes to the code base by simply running the submit command.
Note however that you cannot use any interactive session commands (e.g. python, vim) in this environment, but you can write scripts and run them. E.g. you can write a python script and then run it with `python <script_name>.py`.

NOTE ABOUT THE EDIT COMMAND: Indentation really matters! When editing a file, make sure to insert appropriate indentation before each line!

IMPORTANT TIPS:
1. Always start by trying to replicate the bug that the issues discusses.
   If the issue includes code for reproducing the bug, we recommend that you re-implement that in your environment, and run it to make sure you can reproduce the bug.
   Then start trying to fix it.
   When you think you've fixed the bug, re-run the bug reproduction script to make sure that the bug has indeed been fixed.

   If the bug reproduction script does not print anything when it successfully runs, we recommend adding a print("Script completed successfully, no errors.") command at the end of the file,
   so that you can be sure that the script indeed ran fine all the way through.

2. If you run a command and it doesn't work, try running a different command. A command that did not work once will not work the second time unless you modify it!

3. If you open a file and need to get to an area around a specific line that is not in the first 100 lines, say line 583, don't just use the scroll_down command multiple times. Instead, use the goto 583 command. It's much quicker.

4. If the bug reproduction script requires inputting/reading a specific file, such as buggy-input.png, and you'd like to understand how to input that file, conduct a search in the existing repo code, to see whether someone else has already done that. Do this by running the command: find_file "buggy-input.png" If that doesn't work, use the linux 'find' command.

5. Always make sure to look at the currently open file and the current working directory (which appears right after the currently open file). The currently open file might be in a different directory than the working directory! Note that some commands, such as 'create', open files, so they might change the current  open file.

6. When editing files, it is easy to accidentally specify a wrong line number or to write code with incorrect indentation. Always check the code after you issue an edit to make sure that it reflects what you wanted to accomplish. If it didn't, issue another command to fix it.


(Open file: n/a)
(Current directory: /django-environ)
bash-$
