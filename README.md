# Exploring Methods for Enhancing Appearance-Based Video Object Tracking using Dynamics Theory

This repository corresponds to my M. Sc. Thesis carried out in <a href="http://robustsystems.coe.neu.edu/">Robust Systems Lab</a> at <b>Northeastern University </b> of Boston, in partial fulfilment of the requirements for the <b>BarcelonaTech</b>'s Master in Advanced Telecommunication Technologies.

## Abstract 
The task of Video Object Tracking has for a long time received attention within the field of Computer Vision, and many different approaches have tried to tackle its challenges, being the ones based on appearance and motion some of the most popular ones. The main focus of this thesis is to fuse both approaches in order to exploit their strengths and complement each other's flaws.

To achieve this goal, we propose a unified framework that combines, in an online manner, an off-the-shelfsingle-object siamese tracker (SiamMask), which is modified to perform multi-object tracking and to provide more than one detection candidate, with a novel motion module. This module detects when the proposed target position is not dynamically consistent and, if that is the case, predicts an alternative which is used to choose the best among the rest of candidates.

Our approach is tested on the challenging Similar Multi-Object Tracking Dataset and it achieves an impressing \hl{improvement of 5\%} with respect to the baseline. Although there is still room for improvement mainly regarding the efficiency of the approach, this work has served as a relevant proof of concept for the intuitions behind it and consequently, research in this direction will surely continue at the Robust Systems Laboratory.

## Single Object Tracking Examples
Baseline in red and ours in green.

<img src="/memory/gifs/football_both.gif" width="600" height="335"/>
<img src="/memory/gifs/hockey_both.gif" width="600" height="335"/>
<img src="/memory/gifs/soccer_both.gif" width="600" height="335"/>

## Multiple Object Tracking Examples
Baseline on the left and our approach on the right.

<img src="/memory/gifs/acrobats_siam.gif" width="450" height="260"/> <img src="/memory/gifs/acrobats_ours.gif" width="450" height="260"/>
