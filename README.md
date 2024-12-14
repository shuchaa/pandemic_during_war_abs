# Spatio-Temporal Model of Pandemic Spread During Warfare with Dual-use Healthcare System

## Abstract
Large-scale crises such as war and pandemics have repeated over history. In some cases, the two occur at the same time and place, challenging the affected society. Understanding the dynamics of epidemic spread during warfare is essential for developing effective containment strategies in complex conflict zones. While research has explored epidemic models in various settings, the impact of warfare on epidemic dynamics remains underexplored. In this study, we proposed a novel mathematical model that integrates the epidemiological SIR (susceptible-infected-recovered) model with the war dynamics Lanchester model to explore the dual influence of war and pandemic on a population's mortality. Moreover, we consider a dual-use military and civil healthcare system that aims to reduce the overall mortality rate. Using an agent-based simulation and deep reinforcement learning, we conducted an intensive _in silico_ investigation. Our results show that a pandemic during war conduces chaotic dynamics where the healthcare system should either prioritize war-injured soldiers or pandemic-infected civilians based on the immediate amount of mortality from each option, ignoring long-term objectives. Our findings underscore the need for incorporating conflict-related factors into epidemic modeling to improve preparedness and response strategies in conflict-affected areas.

![fighting soldier](images/war_zone.jpg)


## Table of contents
1. [Code usage](#code_usage)
2. [How to cite](#how_to_cite)
3. [Dependencies](#dependencies)
4. [Contributing](#contributing)
5. [Contact](#contact)


<a name="code_usage"/>

## Code usage
### Run the experiments shown in the paper:
1. Clone the repo 
2. Install the `requirements.txt` file.
3. run the project from the `paper.py` file, it will generate all information needed to the paper and save it under plots folder

```
python paper.py
```

for more execution options see the full usage under
```
python paper.py --help
```

<a name="how_to_cite"/>

## How to cite
Please cite the SciMED work if you compare, use, or build on it:
```
@article{lazebnik2025pandemicwar,
        title={Spatio-Temporal Model of Pandemic Spread During Warfare with Dual-use Healthcare System},
        author={Shuchami, Adi and Givon-Lavi, Noga and Lazebnik, Teddy},
        journal={TBD},
        year={2025}
}
```

<a name="dependencies"/>

## Dependencies 
1. numpy 
2. pandas 
3. networkx 
4. matplotlib
5. tqdm
6. logging
7. colorama
8. torch

<a name="contributing"/>

## Contributing
We would love you to contribute to this project, pull requests are very welcome! Please send us an email with your suggestions or requests...

<a name="contact"/>

## Contact
* Adi Shuchami - [email](mailto:a.shuchami@gmail.com) | [LinkedInֿ](https://www.linkedin.com/in/adi-shuchami-1a93aa7a/)
* Teddy Lazebnik - [email](mailto:lazebnik.teddy@gmail.com) | [LinkedInֿ](https://www.linkedin.com/in/teddy-lazebnik/)

