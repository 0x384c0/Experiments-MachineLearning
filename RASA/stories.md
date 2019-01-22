## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* mood_affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* mood_deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## got error path 1
* greet
  - utter_greet
* got_programm_error
  - utter_so_google_this_error_dont_ask_me

## got error path 2
* got_programm_error
  - utter_so_google_this_error_dont_ask_me