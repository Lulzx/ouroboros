"""Tokenizer for gene regulatory network circuits."""

import json
from pathlib import Path
from typing import Optional


class GRNTokenizer:
    """Tokenizer for encoding/decoding gene regulatory network circuits."""

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # Phenotype tokens
    PHENOTYPE_TOKENS = [
        "<oscillator>",
        "<toggle_switch>",
        "<adaptation>",
        "<pulse_generator>",
        "<amplifier>",
        "<stable>",
    ]

    # Interaction tokens
    INTERACTION_TOKENS = [
        "activates",
        "inhibits",
    ]

    # Default gene vocabulary (canonical genes from synthetic biology and signaling)
    DEFAULT_GENES = [
        # Synthetic biology parts
        "lacI", "tetR", "cI", "araC", "luxR", "lasR",
        # Tumor suppressors and oncogenes
        "p53", "mdm2", "p21", "rb", "myc", "ras", "braf",
        # Signaling kinases
        "erk", "mek", "raf", "akt", "pi3k", "mapk", "jnk", "p38",
        # Transcription factors
        "nfkb", "stat3", "jun", "fos", "creb", "hif1a", "sox2", "oct4",
        # Cell cycle
        "cdk1", "cdk2", "cyclinD", "cyclinE", "cyclinB", "cdc25", "wee1",
        # Apoptosis
        "bcl2", "bax", "casp3", "casp9", "apaf1", "bid", "bad",
        # Growth factors and receptors
        "egfr", "vegf", "tgfb", "wnt", "notch", "shh", "fgf",
        # E. coli regulators
        "crp", "fnr", "arca", "arcb", "ompr", "envz", "narl", "narp",
        # Generic placeholders for flexibility
        "geneA", "geneB", "geneC", "geneD", "geneE", "geneF",
        "geneX", "geneY", "geneZ",
        # Repressors/activators
        "repA", "repB", "repC", "actA", "actB", "actC",
        # Additional signaling
        "src", "jak", "smad", "beta_catenin", "gsk3",
        # Metabolic
        "ampk", "mtor", "sirt1", "foxo",
        # DNA damage
        "atm", "atr", "chk1", "chk2", "brca1",
        # Epigenetic
        "hdac", "dnmt", "ezh2", "brd4",
        # Immune
        "il6", "tnf", "ifng", "il1b", "il10",
        # Yeast genes
        "cln1", "cln2", "clb1", "clb2", "sic1", "cdc14", "cdh1",
    ]

    def __init__(self, genes: Optional[list[str]] = None):
        """Initialize tokenizer with optional custom gene list."""
        self.genes = genes if genes is not None else self.DEFAULT_GENES.copy()

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build token to ID mappings."""
        self.token_to_id = {}
        self.id_to_token = {}

        idx = 0

        # Special tokens (0-3)
        for token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Phenotype tokens (4-15, with room for expansion)
        self.phenotype_start_idx = idx
        for token in self.PHENOTYPE_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        # Reserve space for more phenotypes
        idx = 16

        # Interaction tokens (16-25)
        self.interaction_start_idx = idx
        for token in self.INTERACTION_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        # Reserve space for more interactions
        idx = 26

        # Gene tokens (26+)
        self.gene_start_idx = idx
        for gene in self.genes:
            gene_lower = gene.lower()
            if gene_lower not in self.token_to_id:
                self.token_to_id[gene_lower] = idx
                self.id_to_token[idx] = gene_lower
                idx += 1

        self.vocab_size = idx

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    def get_phenotype_token_ids(self) -> list[int]:
        """Return list of all phenotype class token IDs."""
        return [self.token_to_id[p] for p in self.PHENOTYPE_TOKENS]

    def get_phenotype_name(self, token_id: int) -> str:
        """Get phenotype name from token ID."""
        token = self.id_to_token.get(token_id, "")
        if token.startswith("<") and token.endswith(">"):
            return token[1:-1]  # Remove < and >
        return token

    def phenotype_to_token(self, phenotype: str) -> str:
        """Convert phenotype name to token string."""
        if not phenotype.startswith("<"):
            phenotype = f"<{phenotype}>"
        return phenotype

    def encode(self, circuit: dict) -> list[int]:
        """
        Encode a circuit dictionary to token IDs.

        Args:
            circuit: Dictionary with 'phenotype' and 'interactions' keys.
                    interactions is a list of dicts with 'source', 'target', 'type'.

        Returns:
            List of token IDs: <bos> <phenotype> gene interaction gene ... <eos>
        """
        tokens = [self.bos_token_id]

        # Add phenotype token
        phenotype = circuit.get("phenotype", "stable")
        phenotype_token = self.phenotype_to_token(phenotype)
        tokens.append(self.token_to_id.get(phenotype_token, self.unk_token_id))

        # Add interactions
        interactions = circuit.get("interactions", [])
        for i, interaction in enumerate(interactions):
            source = interaction["source"].lower()
            target = interaction["target"].lower()
            interaction_type = interaction["type"].lower()

            # Add source gene (only for first interaction or if different from previous target)
            if i == 0:
                tokens.append(self.token_to_id.get(source, self.unk_token_id))

            # Add interaction type
            tokens.append(self.token_to_id.get(interaction_type, self.unk_token_id))

            # Add target gene
            tokens.append(self.token_to_id.get(target, self.unk_token_id))

        tokens.append(self.eos_token_id)
        return tokens

    def encode_flat(self, circuit: dict) -> list[int]:
        """
        Encode circuit with each interaction as source-type-target triplets.
        This format is clearer and more flexible.

        Format: <bos> <phenotype> src1 type1 tgt1 src2 type2 tgt2 ... <eos>
        """
        tokens = [self.bos_token_id]

        # Add phenotype token
        phenotype = circuit.get("phenotype", "stable")
        phenotype_token = self.phenotype_to_token(phenotype)
        tokens.append(self.token_to_id.get(phenotype_token, self.unk_token_id))

        # Add each interaction as triplet
        interactions = circuit.get("interactions", [])
        for interaction in interactions:
            source = interaction["source"].lower()
            target = interaction["target"].lower()
            interaction_type = interaction["type"].lower()

            tokens.append(self.token_to_id.get(source, self.unk_token_id))
            tokens.append(self.token_to_id.get(interaction_type, self.unk_token_id))
            tokens.append(self.token_to_id.get(target, self.unk_token_id))

        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int]) -> dict:
        """
        Decode token IDs back to circuit dictionary.

        Args:
            token_ids: List of token IDs

        Returns:
            Circuit dictionary with 'phenotype' and 'interactions'
        """
        # Remove special tokens and get raw tokens
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, self.UNK_TOKEN)
            if token not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                tokens.append(token)

        if not tokens:
            return {"phenotype": "unknown", "interactions": []}

        # First token should be phenotype
        phenotype = "unknown"
        start_idx = 0
        if tokens[0].startswith("<") and tokens[0].endswith(">"):
            phenotype = tokens[0][1:-1]  # Remove < and >
            start_idx = 1

        # Parse interactions (triplet format: source type target)
        interactions = []
        remaining = tokens[start_idx:]

        i = 0
        while i + 2 < len(remaining):
            source = remaining[i]
            interaction_type = remaining[i + 1]
            target = remaining[i + 2]

            # Validate that interaction_type is valid
            if interaction_type in ["activates", "inhibits"]:
                interactions.append({
                    "source": source,
                    "target": target,
                    "type": interaction_type,
                })
                i += 3
            else:
                # Skip malformed tokens
                i += 1

        return {
            "phenotype": phenotype,
            "interactions": interactions,
        }

    def is_valid_circuit(self, token_ids: list[int]) -> bool:
        """
        Check if a circuit is strictly valid.

        A valid circuit must:
        - Have at least one interaction
        - Have source and target be gene tokens (not interaction tokens)
        - Follow gene-interaction-gene triplet pattern
        - Stop at EOS (no trailing garbage)
        - Have exactly N*3 content tokens (complete triplets only)

        Args:
            token_ids: List of token IDs

        Returns:
            True if circuit is valid, False otherwise
        """
        # Get gene set (all tokens that are not special/interaction/phenotype)
        special_tokens = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        interaction_set = set(self.INTERACTION_TOKENS)
        phenotype_set = set(self.PHENOTYPE_TOKENS)

        gene_set = set()
        for token in self.token_to_id.keys():
            if token not in special_tokens and token not in interaction_set and token not in phenotype_set:
                gene_set.add(token)

        # Decode tokens - stop at EOS
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, self.UNK_TOKEN)
            if token == self.EOS_TOKEN:
                break  # Stop at EOS, ignore anything after
            if token not in [self.PAD_TOKEN, self.BOS_TOKEN]:
                tokens.append(token)

        if not tokens:
            return False

        # Skip phenotype token
        start_idx = 0
        if tokens[0].startswith("<") and tokens[0].endswith(">"):
            start_idx = 1

        remaining = tokens[start_idx:]

        # Must have at least 3 tokens for one interaction
        if len(remaining) < 3:
            return False

        # Must have exactly N*3 tokens (complete triplets, no trailing tokens)
        if len(remaining) % 3 != 0:
            return False

        # Check triplet pattern: gene-interaction-gene
        valid_interactions = 0
        for i in range(0, len(remaining), 3):
            source = remaining[i]
            interaction_type = remaining[i + 1]
            target = remaining[i + 2]

            # Source and target must be genes, not interactions or phenotypes
            source_is_gene = source in gene_set
            target_is_gene = target in gene_set
            interaction_valid = interaction_type in interaction_set

            if source_is_gene and target_is_gene and interaction_valid:
                valid_interactions += 1
            else:
                # Invalid pattern
                return False

        return valid_interactions > 0

    def encode_batch(
        self,
        circuits: list[dict],
        max_length: Optional[int] = None,
        flat: bool = True,
    ) -> tuple[list[list[int]], list[int]]:
        """
        Batch encode circuits with padding.

        Args:
            circuits: List of circuit dictionaries
            max_length: Maximum sequence length (pads/truncates to this)
            flat: Use flat encoding format

        Returns:
            Tuple of (padded_sequences, lengths)
        """
        encode_fn = self.encode_flat if flat else self.encode
        sequences = [encode_fn(c) for c in circuits]
        lengths = [len(s) for s in sequences]

        if max_length is None:
            max_length = max(lengths)

        # Pad sequences
        padded = []
        for seq in sequences:
            if len(seq) > max_length:
                padded.append(seq[:max_length])
            else:
                padded.append(seq + [self.pad_token_id] * (max_length - len(seq)))

        return padded, lengths

    def save(self, path: str | Path):
        """Save tokenizer vocabulary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "genes": self.genes,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GRNTokenizer":
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(genes=data["genes"])

    def add_gene(self, gene: str):
        """Add a gene to the vocabulary."""
        gene_lower = gene.lower()
        if gene_lower not in self.token_to_id:
            self.genes.append(gene_lower)
            idx = self.vocab_size
            self.token_to_id[gene_lower] = idx
            self.id_to_token[idx] = gene_lower
            self.vocab_size += 1

    def __len__(self) -> int:
        return self.vocab_size
