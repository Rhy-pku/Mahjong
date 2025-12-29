# MCTS Debug Report

## Summary
- type_counts: {'config': 1, 'episode_start': 1, 'decision_start': 51, 'leaf_eval': 644, 'decision_end': 51, 'env_step': 51, 'env_step_multi': 51, 'simulation_terminal': 188, 'episode_end': 1}
- mcts_decisions: 13
- mcts_action_diff_count: 2
- mcts_action_diff_ratio: 0.15384615384615385
- mcts_select_steps: 624
- u_dom_ratio: 0.5913461538461539
- avg_abs_u: 2.1247811624541497
- avg_abs_q: 1.5400525532089746
- trees_found: 13
- trees_rendered: 13

## Decisions

| step | action | argmax_action | action_is_argmax | root_n_visits | root_valid_count | root_explored_branches | root_prior_max | root_prior_second | root_prior_entropy | tree_dot | tree_png |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 7 | 13 | False | 64 | 12 | 3 | 0.6109364628791809 | 0.3113948702812195 | 1.0151808218182112 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step0.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step0.png |
| 8 | 13 | 13 | True | 64 | 12 | 3 | 0.9947177767753601 | 0.0026139889378100634 | 0.041667134804357915 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step8.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step8.png |
| 16 | 25 | 25 | True | 64 | 11 | 3 | 0.5990877747535706 | 0.2687518000602722 | 1.201088575046781 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step16.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step16.png |
| 24 | 6 | 6 | True | 64 | 12 | 1 | 0.8943203687667847 | 0.0922616720199585 | 0.3950605462053165 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step24.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step24.png |
| 32 | 22 | 22 | True | 64 | 11 | 1 | 0.8373858332633972 | 0.1345735341310501 | 0.5712240495263253 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step32.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step32.png |
| 40 | 31 | 31 | True | 64 | 12 | 2 | 0.5688713192939758 | 0.346471905708313 | 1.0388019722975501 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step40.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step40.png |
| 48 | 26 | 26 | True | 64 | 13 | 1 | 0.9686756134033203 | 0.021381188184022903 | 0.1737515953956599 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step48.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step48.png |
| 56 | 10 | 32 | False | 64 | 13 | 2 | 0.49820420145988464 | 0.4536372721195221 | 0.9310406544750407 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step56.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step56.png |
| 64 | 17 | 17 | True | 64 | 13 | 1 | 0.8427476286888123 | 0.14703842997550964 | 0.49058040619931315 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step64.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step64.png |
| 72 | 33 | 33 | True | 64 | 12 | 2 | 0.5468299984931946 | 0.4312170445919037 | 0.8063761011567065 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step72.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step72.png |
| 80 | 32 | 32 | True | 64 | 13 | 2 | 0.8461257219314575 | 0.1381213366985321 | 0.5104342456580416 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step80.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step80.png |
| 88 | 14 | 14 | True | 64 | 14 | 1 | 0.9827985167503357 | 0.009744980372488499 | 0.11023140672061421 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step88.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step88.png |
| 96 | 32 | 32 | True | 64 | 13 | 1 | 0.9236274361610413 | 0.053229913115501404 | 0.3555651441208674 | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step96.dot | /Users/rhy/Downloads/Chinese-Standard-Mahjong/logs/trees/tree_ep0_step96.png |