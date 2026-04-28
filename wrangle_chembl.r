library(tidyverse)
library(data.table)
library(RPostgres)
library(here)
library(powerjoin)
library(fst)

synapser::synLogin()
syn <- synExtra::synDownloader("~/data", .cache = TRUE)

compound_mapping <- syn("syn26260389") %>%
  read_fst(as.data.table = TRUE)

drv <- dbDriver("Postgres")

con <- dbConnect(
  drv,
  dbname = "chembl_33",
  host = "localhost"
)

# Normalize all units to nM, saving the factors here
standard_unit_map <- c(
  'M' = 1,
  'mol/L' = 1,
  'nM' = 10^-9,
  'nmol/L' = 10^-9,
  'nmol.L-1' = 10^-9,
  'pM' = 10^-12,
  'pmol/L' = 10^-12,
  'pmol/ml' = 10^-9,
  'um' = 10^-6,
  'uM' = 10^-6,
  'umol/L' = 10^-6,
  'umol/ml' = 10^-3,
  'umol/uL' = 1
) %>%
  magrittr::multiply_by(10^9)

approved_standard_types <- c(
  "Kd apparent",
  "Kd",
  "Ki"
)

# Include Klaeger pubmed ID 29191878
kds_raw <- dbGetQuery(
  con,
  paste0(
    "select A.doc_id, ACT.activity_id, A.assay_id, ACT.molregno, MOL_DICT.chembl_id as chembl_id_compound, ACT.standard_relation, ACT.standard_type,
     ACT.standard_value, ACT.standard_units,
     TARGET_DICT.tax_id AS tax_id,
     A.tid,
     A.description,A.chembl_id as chembl_id_assay, BAO.label, DOCS.chembl_id as chembl_id_doc, DOCS.pubmed_id as pubmed_id, TARGET_DICT.organism AS organism,
     CS.accession AS target_accession
     from activities as ACT
     left join assays as A
     on ACT.assay_id = A.assay_id
     LEFT JOIN target_components AS TC
     on A.tid = TC.tid
     LEFT JOIN component_sequences AS CS
     on TC.component_id = CS.component_id
     left join DOCS
     on DOCS.doc_id=A.doc_id
     left join bioassay_ontology as BAO
     on A.bao_format=BAO.bao_id
     LEFT JOIN molecule_dictionary AS MOL_DICT
     on ACT.molregno = MOL_DICT.molregno
     LEFT JOIN target_dictionary AS TARGET_DICT
     on A.tid = TARGET_DICT.tid
     WHERE ACT.standard_value is not null
     AND TARGET_DICT.organism in ('Homo sapiens')
     AND CS.component_type = 'PROTEIN'
     and A.assay_type = 'B'
     and A.relationship_type in ('D', 'H', 'M', 'U')
     and A.bao_format not in ('BAO_0000221', 'BAO_0000219','BAO_0000218')
     and ACT.standard_units in (", paste(paste0("'", names(standard_unit_map), "'"), collapse = ","), ")
     and ACT.standard_type in (", paste(paste0("'", approved_standard_types, "'"), collapse = ","), ")"
  )
)

if (!file.exists("data/HUMAN_9606_idmapping.dat.gz"))
  download.file(
    "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz",
    "data/HUMAN_9606_idmapping.dat.gz",
    method = "curl"
  )

uniprot_map <- fread("data/HUMAN_9606_idmapping.dat.gz", sep = "\t", col.names = c("uniprot_id", "external_db", "external_id"))

kds <- kds_raw %>%
  as_tibble() %>%
  power_left_join(
    uniprot_map %>%
      filter(external_db == "Gene_Name") %>%
      distinct(uniprot_id, gene_symbol = external_id),
    by = c("target_accession" = "uniprot_id"),
    check = check_specs(
      duplicate_keys_right = "warn"
    )
  ) %>%
  power_left_join(
    compound_mapping %>%
      filter(source == "chembl") %>%
      distinct(lspci_id, external_id),
    by = c("chembl_id_compound" = "external_id"),
    check = check_specs(
      duplicate_keys_right = "warn",
      unmatched_keys_left = "warn"
    )
  )

fwrite(
  kds,
  "data/chembl_kds_raw.csv.gz"
)

synExtra::synStoreMany(
  "data/chembl_kds_raw.csv.gz",
  parentId = "syn18508401",
  forceVersion = FALSE,
  used = c(
    "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz",
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_postgresql.tar.gz",
    "syn26260389"
  ),
  executed = "https://github.com/labsyspharm/okl-analysis/blob/main/wrangle_chembl.Rmd"
)

