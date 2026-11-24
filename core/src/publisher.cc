#include "publisher.h"

PublisherPtr MakeNullPublisher() {
  static PublisherPtr publisher = std::make_shared<NullPublisher>();
  return publisher;
}
