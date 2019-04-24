(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(clear t0)
		(darkblue b0)
		(blue b0)
		(firebrick b1)
		(red b1)
		(bisque b2)
		(orange b2)
		(indigo b3)
		(purple b3)
		(darkred b4)
		(red b4)
		(darkorange b5)
		(orange b5)
		(darkviolet b6)
		(purple b6)
		(darkorchid b7)
		(purple b7)
		(royalblue b8)
		(blue b8)
		(darkorange b9)
		(orange b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (forall (?y) (or (not (orange ?y)) (exists (?x) (and (purple ?x) (on ?x ?y)))))))
)