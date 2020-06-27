# Exploring Methods for Enhancing Appearance-Based Video Object Tracking using Dynamics Theory

This repository corresponds to my M. Sc. Thesis carried out in <a href="http://robustsystems.coe.neu.edu/">Robust Systems Lab</a> at <b>Northeastern University </b> of Boston, in partial fulfilment of the requirements for the <b>BarcelonaTech</b>'s Master in Advanced Telecommunication Technologies.

## Abstract 
The task of Video Object Tracking has for a long time received attention within the field of Computer Vision, and many different approaches have tried to tackle its challenges, being the ones based on appearance and motion some of the most popular ones. The main focus of this thesis is to fuse both approaches in order to exploit their strengths and complement each other's flaws.

To achieve this goal, we propose a unified framework that combines, in an online manner, an <i>off-the-shelf</i single-object siamese tracker (SiamMask), which is modified to perform multi-object tracking and to provide more than one detection candidate, with a Dynamics Module (DM). This module detects when the proposed target position is not dynamically consistent and, if that is the case, predicts an alternative which is used to choose the best among the rest of candidates.

Our approach is evaluated on the challenging Similar Multi-Object Tracking (SMOT) dataset and it achieves an impressing precision improvement of the 10% with respect to the baseline. We present an extension to the SMOT dataset, the eSMOT, including more sequences with complex dynamic scenarios, where the performance of our approach is excellent, therefore we use its predictions to label the Ground Truth. 

Although there is still room for enhancement mainly regarding the efficiency of the architecture, this work has served as a relevant proof of concept for the intuitions behind it and consequently, research in this direction will surely continue at the Robust Systems Laboratory.

## Single Object Tracking Examples
Baseline in <b>red</b> and ours in <b>green</b>, sequences <i>football</i>, <i>hockey</i> and <i>soccer</i> from eSMOT dataset.

<img src="/figures/gifs/football_both.gif" width="600" height="335"/>
<img src="/figures/gifs/hockey_both.gif" width="600" height="335"/>
<img src="/figures/gifs/soccer_both.gif" width="600" height="335"/>

## Multiple Object Tracking Example
Baseline on the <b>left</b> and our approach on the  <b>right</b>, sequence  <i>acrobats</i> from SMOT dataset.

<img src="/figures/gifs/acrobats_siam.gif" width="400" height="220"/> <img src="/figures/gifs/acrobats_ours.gif" width="400" height="220"/>
