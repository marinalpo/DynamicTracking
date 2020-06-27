# Exploring Methods for Enhancing Appearance-Based Video Object Tracking using Dynamics Theory

The task of Video Object Tracking has for a long time received attention within the field of Computer Vision, and many different approaches have tried to tackle its challenges, being the ones based on appearance and motion some of the most popular ones. The main focus of this thesis is to fuse both approaches in order to exploit their strengths and complement each other's flaws.

To achieve this goal, we propose a unified framework that combines, in an online manner, an \textit{off-the-shelf} single-object siamese tracker, which is modified to perform multi-object tracking and to provide more than one detection candidate, with a novel motion module. This module detects when the proposed target position is not dynamically consistent and, if that is the case, predicts an alternative which is used to choose the best among the rest of candidates.

Our approach is tested on the challenging Similar Multi-Object Tracking Dataset and it achieves an impressing \hl{improvement of 5\%} with respect to the baseline. Although there is still room for improvement mainly regarding the efficiency of the approach, this work has served as a relevant proof of concept for the intuitions behind it and consequently, research in this direction will surely continue at the Robust Systems Laboratory.


