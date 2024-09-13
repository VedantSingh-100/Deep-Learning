from itertools import groupby
import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        symbol_length, seq_len, batch_size = y_probs.shape

        for t in range(seq_len):
            probs_t = y_probs[:, t, :]
            max_idx = np.argmax(probs_t)
            max_prob = probs_t[max_idx]
            if max_idx == 0:
                decoded_path.append('BLANK')
            else:
                decoded_path.append(self.symbol_set[max_idx - 1])
            path_prob *= max_prob

        compressed_path = [decoded_path[0]]
        for i in range(1, len(decoded_path)):
            if decoded_path[i] != decoded_path[i - 1]:
                compressed_path.append(decoded_path[i])

        compressed_path = np.array(compressed_path)
        decoded_path = "".join(compressed_path[compressed_path != 'BLANK'])

        return decoded_path, path_prob
        # raise NotImplementedError


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        sequence_len = y_probs.shape[1]
        updated_symbols = self.symbol_set
        updated_symbols.insert(0, "-")
        top_paths = {"-": 1.0}

        for t_step in range(sequence_len):
            updated_top_paths = {}

            symbol_probs = y_probs[:, t_step, 0]

            for current_seq, current_seq_prob in top_paths.items():
                for i in range(len(symbol_probs)):
                    sym = updated_symbols[i]
                    last_sym = current_seq[-1]

                    if last_sym == sym:
                        new_seq = current_seq
                    elif last_sym == "-":
                        new_seq = current_seq[:-1] + sym
                    else:
                        new_seq = current_seq + sym

                    updated_top_paths[new_seq] = updated_top_paths.get(new_seq, 0) + current_seq_prob * symbol_probs[i]

            sorted_updated_top_paths = sorted(updated_top_paths.items(), key=lambda item: item[1], reverse=True)

            top_paths = dict(sorted_updated_top_paths[:self.beam_width])

        aggregated_path_scores = {}

        for seq, prob in sorted_updated_top_paths:
            if seq[-1] == "-":
                seq = seq[:-1]

            aggregated_path_scores[seq] = aggregated_path_scores.get(seq, 0) + prob

        optimal_sequence = max(aggregated_path_scores, key=aggregated_path_scores.get)

        return optimal_sequence, aggregated_path_scores