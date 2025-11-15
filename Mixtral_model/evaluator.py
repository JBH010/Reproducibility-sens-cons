"""
       Sensitivity and Consistency of Large Language Models

  File:     evaluator.py
  Authors:  Federico Errica (federico.errica@neclab.eu)
            Giuseppe Siracusano (giuseppe.siracusano@neclab.eu)
	    Davide Sanvito (davide.sanvito@neclab.eu)
	    Roberto Bifulco (roberto bifulco@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2025-, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicabl"""
"""
Evaluator module for running grid search experiments with LLMs."""

"""
Evaluator module for running grid search experiments with LLMs.
"""
import sys
import json
import os
import time
from pathlib import Path

"""
Evaluator module for running grid search experiments with LLMs.
Optimized to only generate answers on HPC - process results locally later.
"""

import sys
import json
import os
import time
from pathlib import Path


def run_grid_search(
    samples,
    llms,
    prompt_types,
    Qs,
    temp_questions,
    question_to_rewrite,
    As,
    temp_answers,
    class_labels,
    class_extractor_fun,
    results_folder,
    num_actors=4,
):
    """
    Run Grid Search on the given samples with proper Ray parallelization.
    
    Args:
        samples: List of samples to process
        llms: List of LLM names
        prompt_types: Dictionary of prompt type names to prompts
        Qs: List of number of question rephrasings
        temp_questions: List of temperatures for question generation
        question_to_rewrite: The question to rephrase
        As: List of number of answer calls
        temp_answers: List of temperatures for answers
        class_labels: List of class labels
        class_extractor_fun: Function to extract class from text
        results_folder: Path to save results
        num_actors: Number of Ray actors (GPUs) to use for parallel processing
    """
    print("→ Entered run_grid_search function!", flush=True)
    sys.stdout.flush()
    
    print("→ Importing torch...", flush=True)
    sys.stdout.flush()
    import torch
    print("✓ torch imported", flush=True)
    sys.stdout.flush()
    
    print("→ Importing ray...", flush=True)
    sys.stdout.flush()
    import ray
    print("✓ ray imported", flush=True)
    sys.stdout.flush()
    
    print("→ Importing model_actor...", flush=True)
    sys.stdout.flush()
    from model_actor import LLMModelActor
    print("✓ model_actor imported", flush=True)
    sys.stdout.flush()
    
    # Initialize Ray ONCE at the start
    print("→ Initializing Ray...", flush=True)
    sys.stdout.flush()
    ray.shutdown()  # Clean up any existing instance

    # Detect available GPUs
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected {num_gpus_available} physical GPUs", flush=True)

    # Don't limit num_actors - allow fractional GPU allocation
    print(f"Will use {num_actors} actor(s) (with fractional GPU allocation)", flush=True)

    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        num_cpus=120,  # Adjust based on your system
        num_gpus=num_gpus_available,
    )
    print("Ray initialized successfully", flush=True)

    try:
        # CREATE ACTORS ONCE PER LLM - OUTSIDE ALL LOOPS
        for llm in llms:
            print(f"\n{'='*80}")
            print(f"Creating {num_actors} Model Actors for {llm}...")
            print(f"{'='*80}\n", flush=True)

            model_actors = [
                LLMModelActor.remote(
                    llm=llm,
                    temperature_question=0.0,  # Default value
                    num_questions=30,  # Max value we'll use
                    llm_rephraser=None,
                )
                for _ in range(num_actors)
            ]
            print(f"Created {num_actors} Model Actors", flush=True)

            # Wait for all actors to be ready
            print("Waiting for all actors to finish loading models...", flush=True)
            ray.get([actor.ready.remote() for actor in model_actors])
            print("✓ All actors ready!\n", flush=True)

            # NOW LOOP THROUGH PROMPT TYPES WITH SAME ACTORS
            for prompt_type, prompt_original in prompt_types.items():
                data_dict = {}
                filename = f"results_{prompt_type}.json"

                print(f"\n{'='*80}")
                print(f"Processing prompt type: {prompt_type}")
                print(f"{'='*80}\n", flush=True)

                if os.path.exists(Path(results_folder, filename)):
                    with open(Path(results_folder, filename), "r") as f:
                        data_dict = json.load(f)

                if "mixtral" in llm:  # move "system" text to "user"
                    prompt = prompt_original[1:]
                    prompt[0][1] = f"{prompt_original[0][1]} {prompt[0][1]}"
                    print(prompt)
                else:
                    prompt = prompt_original

                for Q in Qs:
                    for temp_question in temp_questions:
                        key_questions = f"{llm}_{Q}_{temp_question}"
                        simple_filename = f"results_simple.json"

                        # Generate questions using first actor (only needs to be done once)
                        if (
                            prompt_type != "simple"
                            and os.path.exists(Path(results_folder, simple_filename))
                            and key_questions not in data_dict
                        ):
                            print('Reusing same prompt variants from "simple" approach...')
                            with open(Path(results_folder, simple_filename), "r") as f:
                                simple_data_dict = json.load(f)
                                alt_questions = simple_data_dict[key_questions]
                                data_dict[key_questions] = alt_questions
                                json_data = json.dumps(data_dict)
                                with open(Path(results_folder, filename), "w") as f:
                                    f.write(json_data)
                        else:
                            if key_questions not in data_dict:
                                print(f"Generating {Q} alternative questions...", flush=True)
                                # Use first actor to generate questions
                                alt_questions = ray.get(
                                    model_actors[0].generate_questions.remote(
                                        question=question_to_rewrite
                                    )
                                )
                                print(f"Generated {len(alt_questions)} questions", flush=True)

                                data_dict[key_questions] = alt_questions
                                json_data = json.dumps(data_dict)
                                with open(Path(results_folder, filename), "w") as f:
                                    f.write(json_data)
                            else:
                                alt_questions = data_dict[key_questions]

                        # Helper function to replace questions locally
                        def replace_question_local(prompt, old_q, new_q):
                            import copy
                            modified = copy.deepcopy(prompt)
                            for message in modified:
                                if len(message) > 1 and isinstance(message[1], str):
                                    message[1] = message[1].replace(old_q, new_q)
                            return modified

                        # Create modified prompts locally (no GPU needed)
                        modified_prompts = [
                            replace_question_local(prompt, question_to_rewrite, alt_question)
                            for alt_question in alt_questions
                        ]

                        for A in As:
                            for temp_answer in temp_answers:
                                # Process in small batches to avoid overwhelming Ray
                                batch_size = 20  # Process 10 samples at a time
                                total_samples = len(samples)
                                completed_count = 0

                                for batch_start in range(0, total_samples, batch_size):
                                    batch_end = min(batch_start + batch_size, total_samples)
                                    batch_samples = samples[batch_start:batch_end]

                                    print(f"\nProcessing batch {batch_start//batch_size + 1}: samples {batch_start}-{batch_end-1}", flush=True)

                                    futures = []
                                    sample_info = []

                                    for sample_idx, sample in enumerate(batch_samples):
                                        s_id = sample["id"]
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        if key in data_dict:
                                            print(f"Skipping test {key}")
                                            continue

                                        # Round-robin: distribute samples across actors
                                        global_idx = batch_start + sample_idx
                                        actor_idx = global_idx % num_actors
                                        actor = model_actors[actor_idx]

                                        print(f"Submitting job {key} to actor {actor_idx}", flush=True)

                                        future = actor.generate_answers.remote(
                                            modified_prompts,
                                            sample["input"],
                                            temp_answer,
                                            A,
                                        )
                                        futures.append(future)
                                        sample_info.append((key, sample["class"]))

                                    if not futures:
                                        print("All samples in this batch already completed, skipping...")
                                        continue

                                    print(f"Waiting for {len(futures)} samples in this batch...", flush=True)

                                    # Wait for ALL in this batch before moving to next batch
                                    for idx, future in enumerate(futures):
                                        key, target = sample_info[idx]

                                        try:
                                            print(f"Waiting for {key}... (timeout: 30 min)", flush=True)
                                            answers = ray.get(future, timeout=1800)  # 30 min timeout per sample
                                            print(f"✓ Got answers for {key}", flush=True)

                                            # OPTIMIZATION: Just save raw answers - process later!
                                            # Convert answers dict to serializable format
                                            serializable_answers = {}
                                            for ans_key, ans_value in answers.items():
                                                if hasattr(ans_value, 'content'):
                                                    serializable_answers[ans_key] = {'content': ans_value.content}
                                                else:
                                                    serializable_answers[ans_key] = {'content': str(ans_value)}

                                            s_id, llm_name, num_q, num_a, tmp_q, tmp_a = key.split("_")
                                            data_dict[key] = dict(
                                                id=s_id,
                                                llm=llm_name,
                                                Q=int(num_q),
                                                A=int(num_a),
                                                target=target,
                                                temp_question=float(tmp_q),
                                                temp_answer=float(tmp_a),
                                                raw_answers=serializable_answers,  # Save raw answers only!
                                            )

                                            completed_count += 1
                                            print(f"✓ Saved raw answers for {key} ({completed_count}/{total_samples})", flush=True)

                                        except ray.exceptions.GetTimeoutError:
                                            print(f"✗ TIMEOUT for {key} after 30 minutes!", flush=True)
                                            continue
                                        except Exception as e:
                                            print(f"✗ ERROR processing {key}: {e}", flush=True)
                                            import traceback
                                            traceback.print_exc()
                                            continue

                                    # Save after each batch
                                    json_data = json.dumps(data_dict)
                                    with open(Path(results_folder, filename), "w") as f:
                                        f.write(json_data)
                                    print(f"💾 Saved batch {batch_start//batch_size + 1} ({completed_count}/{total_samples} total completed)\n", flush=True)

    except Exception as e:
        print(f"ERROR in grid search: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("Shutting down Ray...", flush=True)
        ray.shutdown()
