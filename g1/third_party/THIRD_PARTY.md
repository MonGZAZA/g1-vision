# Third-party sources in this folder

| Folder | Origin | Commit | Licence | Used for |
|---|---|---|---|---|
| `unitree_sdk2_python` | copy of exp04/g1/third_party (https://github.com/unitreerobotics/unitree_sdk2_python.git) | `814556d15970dd2ecf1c9984e845ca02ab07e206` | BSD-3-Clause | Python SDK (DDS topics); installed editable by `uv sync` |
| `unitree_ros/robots/g1_description` (`g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf` + the meshes it uses, downloaded 2026-11-10) | https://github.com/unitreerobotics/unitree_ros | `725b59b6de39a26c425632120c89f23c49a36f1e` | BSD-3-Clause | G1 29-DoF + Inspire RH56DFQ hands; exp05's model is built from it by `tools/build_model.py` |
| `unitree_g1_dex3` (`g1_29dof_with_hand_rev_1_0.urdf` + the meshes it uses, copied 2026-11-10 from exp02/g1/third_party/unitree_rl_gym/resources/robots/g1_description) | https://github.com/unitreerobotics/unitree_rl_gym | `276801e46c5d433564f24658bac64f254b7d2d4b` | BSD-3-Clause | G1 29-DoF + Dex3-1 3-finger hands; `tools/build_model.py --hands dex3` builds `model/g1_dex3.xml` from it |
