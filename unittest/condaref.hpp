#pragma once

#include <memory>
#include <vector>

/** @brief The reference's parameters: points, dimension, R, insert L, alpha and C */
struct CondaSetup {
  unsigned count;
  unsigned width;
  unsigned range;
  unsigned listn;
  float alpha;
  unsigned prune;
};

/** @brief condaann's own in-memory CONDA behind a header free of its types, so its headers never
 *  meet PipeANN's in one translation unit */
class CondaRef {
 public:
  /**
   * @brief An empty CONDA index over the rows of data, tag i at row i
   * @param setup the parameters
   * @param data count rows of width floats
   */
  CondaRef(const CondaSetup &setup, const float *data);

  /** @brief Frees the index */
  ~CondaRef();

  /**
   * @brief BatchInsert_crng of the rows of the given tags
   * @param tags the tags, in insert order
   */
  void insert(const std::vector<unsigned> &tags);

  /**
   * @brief A delete step: MarkDelete of every tag, MultiProcess_crng, ConsolidateDelete_conda
   * @param tags the tags, in MarkDelete order
   */
  void remove(const std::vector<unsigned> &tags);

  /**
   * @brief The neighbor list at a location
   * @param location the location
   * @return its out-neighbors
   */
  std::vector<unsigned> edges(unsigned location) const;

  /**
   * @brief The location of a tag
   * @param tag the tag
   * @return its location, UINT_MAX when absent
   */
  unsigned locate(unsigned tag) const;

  /**
   * @brief CONDA's entry point
   * @return its location
   */
  unsigned entry() const;

  /**
   * @brief Locations ever handed out, CONDA's max_location_
   * @return their count
   */
  unsigned locations() const;

 private:
  struct Inner;
  std::unique_ptr<Inner> inner;
};
