from os.path import join
from pathlib import Path

# Directories
PROJECT_DIRECTORY = Path(__file__).parents[1]
SRC_DIRECTORY = join(PROJECT_DIRECTORY, "src")
INPUT_DIRECTORY = join(PROJECT_DIRECTORY, "input")
RESULT_DIRECTORY = join(PROJECT_DIRECTORY, "results")
CLASS_DIRECTORY = join(SRC_DIRECTORY, "classes")
INTERMEDIATE_DIRECTORY = join(SRC_DIRECTORY, "intermediate_results")
WORD_LIST_DIRECTORY = join(INPUT_DIRECTORY, "defined_word_lists")
GDPR_REA_SPEZIFICATION_DIRECTORY = join(WORD_LIST_DIRECTORY, "gdpr")
ISO_REA_SPEZIFICATION_DIRECTORY = join(WORD_LIST_DIRECTORY, "iso")
ISO_REA_SPEZIFICATION1 = join(ISO_REA_SPEZIFICATION_DIRECTORY, "rea_specification_information_security_management_system.txt")
ISO_REA_SPEZIFICATION2 = join(ISO_REA_SPEZIFICATION_DIRECTORY, "rea_specification_top_management.txt")
ISO_SIGNALWORDS = join(ISO_REA_SPEZIFICATION_DIRECTORY, "signalwords.txt")
GDPR_REA_SPEZIFICATION1 = join(GDPR_REA_SPEZIFICATION_DIRECTORY, "rea_specification_controller.txt")
GDPR_REA_SPEZIFICATION2 = join(GDPR_REA_SPEZIFICATION_DIRECTORY, "rea_specification_data_protection_officer.txt")
GDPR_REA_SPEZIFICATION3 = join(GDPR_REA_SPEZIFICATION_DIRECTORY, "rea_specification_management.txt")
GDPR_SIGNALWORDS = join(GDPR_REA_SPEZIFICATION_DIRECTORY, "signalwords.txt")
GDPR_REG_TEST = join(GDPR_REA_SPEZIFICATION_DIRECTORY, "test_reg.txt")
GDPR_REA_TEST = join(GDPR_REA_SPEZIFICATION_DIRECTORY, "test_rea.txt")